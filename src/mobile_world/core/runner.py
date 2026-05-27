import os
import random
import threading
import time
from queue import Queue

from dotenv import load_dotenv
from joblib import Parallel, delayed
from loguru import logger

from mobile_world.agents.base import BaseAgent, MCPAgent
from mobile_world.agents.checker import Checker, CheckerResult
from mobile_world.agents.mas_divergence import run_mas_divergence
from mobile_world.agents.registry import create_agent
from mobile_world.agents.utils.visual_diff import (
    compute_dhash,
    hamming_distance,
    is_no_change,
)
from mobile_world.runtime.client import (
    AndroidEnvClient,
    AndroidMCPEnvClient,
    scan_finished_tasks,
)
from mobile_world.runtime.utils.docker import (
    discover_backends,
)
from mobile_world.runtime.utils.models import ANSWER, ENV_FAIL, FINISHED, UNKNOWN, JSONAction
from mobile_world.runtime.utils.trajectory_logger import TrajLogger

load_dotenv()


def _should_check_terminal_action(action, enable_answer_trigger: bool = True) -> bool:
    """Return True if the action is a terminal action that the checker should verify.

    By default checks both FINISHED and ANSWER. Set enable_answer_trigger=False to
    only check FINISHED (legacy behavior).
    """
    if action.action_type == FINISHED:
        return True
    if enable_answer_trigger and action.action_type == ANSWER:
        return True
    return False


def _format_checker_constraint(
    checker_result: CheckerResult,
    trigger_reason: str = "completion_check",
) -> str:
    """Build the constraint message that gets injected into the next agent step."""
    return (
        "SYSTEM CONSTRAINT FROM VISUAL CHECKER:\n"
        "Your previous attempt to stop was not visually verified.\n"
        f"Checker reason: {checker_result.reason}\n"
        "You must not terminate successfully until the current screenshot clearly supports "
        "task completion. Continue acting or verifying the state."
    )


def _drop_last_agent_prediction_if_supported(agent) -> bool:
    drop_fn = getattr(agent, "drop_last_prediction_from_history", None)
    if drop_fn is None:
        return False
    return bool(drop_fn())


def _execute_single_task(
    env: AndroidEnvClient,
    agent: BaseAgent,
    task_name: str,
    max_step: int,
    traj_logger: TrajLogger,
    enable_mcp: bool = False,
    checker: Checker | None = None,
    dhash_threshold: int = 4,
    no_change_k: int = 3,
    enable_answer_trigger: bool = True,
    mas_enabled: bool = False,
    mas_skip_recent_k: int = 8,
    mas_recent_window: int = 5,
    mas_click_tol: int = 80,
    mas_drag_tol: int = 100,
    mas_finished_loop_k: int = 3,
) -> tuple[int, float]:
    """Execute a single task and return the number of steps and score.

    Returns:
        tuple[int, float]: (number of steps, score)
    """

    logger.debug(f"max_step: {max_step}")

    if enable_mcp and not isinstance(agent, MCPAgent):
        logger.error(
            "MCP is enabled but agent type is not a MCP agent. Please use a MCP agent type."
        )

    if enable_mcp:
        traj_logger.log_tools(env.tools)
    task_goal = env.get_task_goal(task_type=task_name)

    logger.debug(f"task_goal: {task_goal}")

    step = 0
    obs = env.initialize_task(task_name=task_name)
    agent.initialize(task_goal)

    # Per-step state for trigger conditions
    pending_checker_constraint: str | None = None
    prev_hash: list[int] | None = None
    no_change_count: int = 0
    last_step_blocked: bool = False
    executed_actions: list[dict] = []
    recent_visual_steps: list[dict] = []
    consecutive_fallback_count: int = 0

    while True:
        step += 1

        logger.debug(f"Screenshot captured in step {step}")

        # ============ Phase 1: visual diff signal ============
        current_hash: list[int] | None = None
        current_distance: int | None = None
        try:
            current_hash = compute_dhash(obs.screenshot)
        except Exception as e:
            logger.warning(f"dHash computation failed at step {step}, skipping signal: {e}")
            current_hash = None

        if prev_hash is not None and current_hash is not None:
            current_distance = hamming_distance(prev_hash, current_hash)
            if last_step_blocked:
                # action was not executed last step (checker blocked it, no MAS override),
                # so screen unchanged is expected. Reset count.
                no_change_count = 0
            elif is_no_change(prev_hash, current_hash, threshold=dhash_threshold):
                no_change_count += 1
            else:
                no_change_count = 0
        # step 1: prev_hash is None → distance stays None, count stays 0

        # ============ Phase 2: agent prediction ============
        agent_observation = {
            "screenshot": obs.screenshot,
            "tool_call": obs.tool_call,
            "ask_user_response": obs.ask_user_response,
        }
        if pending_checker_constraint:
            agent_observation["checker_result"] = pending_checker_constraint
            pending_checker_constraint = None

        prediction, action = agent.predict(agent_observation)

        # ============ Phase 3: trigger decision ============
        # completion_check (FINISHED/ANSWER) → checker (preserved behavior)
        # stuck_loop (no_change_count >= K) → MAS hard override, no checker
        # Triggers are mutually exclusive; completion_check takes priority.
        trigger_reason: str | None = None
        if prediction is not None:
            if _should_check_terminal_action(action, enable_answer_trigger=enable_answer_trigger):
                trigger_reason = "completion_check"
            elif mas_enabled and no_change_count >= no_change_k:
                trigger_reason = "stuck_loop"

        # ============ Phase 4: checker call (only for completion_check) ============
        checker_result = None
        checker_blocked = False
        if checker is not None and prediction is not None and trigger_reason == "completion_check":
            checker_result_obj = checker.check_finished(
                task_goal=task_goal,
                screenshot=obs.screenshot,
                prediction=prediction,
                action=action.model_dump(exclude_none=True),
                step=step,
                recent_steps=recent_visual_steps[-3:],
                trigger_reason=trigger_reason,
            )
            checker_result = checker_result_obj.model_dump()
            checker_blocked = not checker_result_obj.res

        # ============ Phase 5: action determination ============
        # Two independent paths:
        #   (a) checker_blocked (only completion_check): mask + constraint, no execute
        #   (b) trigger_reason == "stuck_loop": MAS divergence, hard override
        action_source = "agent"
        action_to_execute = action
        mas_dump: dict | None = None
        mas_active = False

        if checker_blocked:
            # Checker feedback is advisory/masking only.
            pending_checker_constraint = _format_checker_constraint(
                checker_result_obj,
                trigger_reason=trigger_reason,
            )
            dropped_history = _drop_last_agent_prediction_if_supported(agent)
            logger.info(
                "Checker blocked action in step {} via {}: {} (dropped_history={})",
                step,
                trigger_reason,
                checker_result_obj.reason,
                dropped_history,
            )
        elif trigger_reason == "stuck_loop":
            # MAS divergence (decoupled from checker)
            logger.info("MAS divergence engaging at step {} via stuck_loop trigger", step)
            try:
                mas_result = run_mas_divergence(
                    agent=agent,
                    observation=agent_observation,
                    task_goal=task_goal,
                    executed_actions=executed_actions,
                    skip_recent_k=mas_skip_recent_k,
                    recent_sig_window=mas_recent_window,
                    click_tol=mas_click_tol,
                    drag_tol=mas_drag_tol,
                    rejected_action=action.model_dump(exclude_none=True),
                    consecutive_fallback_count=consecutive_fallback_count,
                )
                mas_dump = mas_result.model_dump()
                selected = dict(mas_result.selected_action)
                action_to_execute = JSONAction(**selected)
                action_source = f"mas_{mas_result.selected_role}"
                mas_active = True
                # Clean main agent's history so it does not retain the rejected
                # prediction that was actually overridden.
                _drop_last_agent_prediction_if_supported(agent)
                if mas_result.fallback_used:
                    consecutive_fallback_count += 1
                else:
                    consecutive_fallback_count = 0
                logger.info(
                    "MAS selected role={} action={} fallback={}",
                    mas_result.selected_role,
                    mas_result.selected_action,
                    mas_result.fallback_used,
                )
            except Exception as exc:
                logger.exception(f"MAS divergence raised at step {step}: {exc}")
                mas_active = False
            # Reset no_change_count after stuck_loop fire (regardless of MAS
            # success) so trigger does not re-fire every subsequent step on
            # the same accumulated count.
            no_change_count = 0

        # ============ Phase 6: persist ============
        traj_logger.log_traj(
            task_name,
            task_goal,
            step,
            prediction,
            action.model_dump(exclude_none=True),
            obs,
            agent.get_total_token_usage(),
            checker_result=checker_result,
            dhash_distance=current_distance,
            no_change_count=no_change_count,
            trigger_reason=trigger_reason,
            mas_divergence=mas_dump,
            action_source=action_source,
        )

        # ============ Phase 7: update state ============
        if current_hash is not None:
            prev_hash = current_hash
        recent_visual_steps.append({"step": step, "screenshot": obs.screenshot})
        # MAS-executed steps have their action take effect, so are NOT
        # considered "blocked" for dHash purposes.
        last_step_blocked = checker_blocked and not mas_active

        if prediction is None:
            logger.warning(f"Agent prediction failed in step {step}")
            break

        # ============ Phase 8: execute action or terminate ============
        terminate = False
        logger.debug(f"current step {step}")

        if mas_active:
            # MAS override: execute the MAS-selected action (never terminal).
            logger.debug(f"executing MAS-overridden action ({action_source}): {action_to_execute}")
            try:
                obs = env.execute_action(action_to_execute)
                executed_actions.append(action_to_execute.model_dump(exclude_none=True))
            except Exception as exc:
                logger.exception(f"MAS-selected action execution failed at step {step}: {exc}")
                raise
        elif checker_blocked:
            # checker block: do not execute the rejected terminal action
            logger.debug("checker blocked action in step {}, continuing without execution", step)
        elif action.action_type in [ENV_FAIL, FINISHED, UNKNOWN]:
            logger.debug(f"task terminated in step {step} with action {action.action_type}")
            terminate = True
        elif action.action_type in [ANSWER]:
            logger.debug(f"answer triggered, execution action {action}")
            obs = env.execute_action(action)
            executed_actions.append(action.model_dump(exclude_none=True))
            terminate = True
        else:
            logger.debug(f"execution action {action}")
            obs = env.execute_action(action)
            executed_actions.append(action.model_dump(exclude_none=True))

        if terminate:
            break

        if step >= max_step:
            logger.debug("task steps reach max step, terminate")
            break

    score, reason = env.get_task_score(task_type=task_name)
    logger.debug(f"task_score: {score}, reason: {reason}")
    traj_logger.log_score(score=score, reason=reason)

    res = env.tear_down_task(task_type=task_name)
    agent.done()
    logger.debug(f"tear_down_task response: {res}")

    return step, score


def _process_task_on_env(
    task_name: str,
    env_queue: Queue,
    agent_type: str,
    model_name: str,
    llm_base_url: str,
    api_key: str | None,
    log_file_root: str,
    max_step: int,
    retry_on_device_unhealthy: int = 2,
    enable_mcp: bool = False,
    checker_enabled: bool = False,
    checker_model_name: str | None = None,
    checker_base_url: str | None = None,
    checker_api_key: str | None = None,
    checker_dhash_threshold: int = 4,
    checker_no_change_k: int = 3,
    checker_enable_answer_trigger: bool = True,
    mas_enabled: bool = False,
    mas_skip_recent_k: int = 8,
    mas_recent_window: int = 5,
    mas_click_tol: int = 80,
    mas_drag_tol: int = 100,
    mas_finished_loop_k: int = 3,
    **kwargs,
) -> dict:
    """Process a single task on a specific environment.

    Args:
        task_name: Name of the task to execute
        env_url: URL of the environment to use
        agent_type: Type of agent to create
        model_name: Model name for the agent
        llm_base_url: LLM service base URL
        api_key: API key for LLM service
        log_file_root: Root directory for log files
        max_step: Maximum steps for task execution
        **kwargs: Additional kwargs for agent creation

    Returns:
        dict: Task result containing task_name, success, score, steps, duration_seconds
    """
    # Create thread-specific log file
    thread_id = threading.current_thread().ident
    thread_log_file = os.path.join(log_file_root, task_name, f"thread_{thread_id}.log")
    os.makedirs(os.path.dirname(thread_log_file), exist_ok=True)
    traj_logger = TrajLogger(log_file_root, task_name)

    def thread_filter(record):
        return record["extra"].get("thread_id") == thread_id

    thread_handler_id = logger.add(
        thread_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | container: {extra[container_name]} | {message}",
        level="DEBUG",
        enqueue=True,
        filter=thread_filter,
    )
    env, container_name = env_queue.get()

    try:
        with logger.contextualize(thread_id=thread_id, container_name=container_name):
            logger.info("Processing task '{}' on environment {}", task_name, env.base_url)
            if enable_mcp:
                assert isinstance(env, AndroidMCPEnvClient), (
                    f"env must be a AndroidMCPEnvClient, but got {type(env)}"
                )
                try:
                    env.reset_tools(task_type=task_name)
                except Exception as e:
                    logger.exception(f"Error resetting tools for task {task_name}: {e}")
                    return None

            agent = create_agent(agent_type, model_name, llm_base_url, api_key, env=env, **kwargs)
            checker = None
            if checker_enabled:
                checker = Checker(
                    model_name=checker_model_name or model_name,
                    base_url=checker_base_url or llm_base_url,
                    api_key=checker_api_key if checker_api_key is not None else api_key,
                )
                logger.info(
                    "Checker enabled: model={} base_url={}",
                    checker.model_name,
                    checker_base_url or llm_base_url,
                )

            task_start_time = time.time()
            while True:
                try:
                    task_steps, task_score = _execute_single_task(
                        env,
                        agent,
                        task_name,
                        max_step,
                        traj_logger=traj_logger,
                        enable_mcp=enable_mcp,
                        checker=checker,
                        dhash_threshold=checker_dhash_threshold,
                        no_change_k=checker_no_change_k,
                        enable_answer_trigger=checker_enable_answer_trigger,
                        mas_enabled=mas_enabled,
                        mas_skip_recent_k=mas_skip_recent_k,
                        mas_recent_window=mas_recent_window,
                        mas_click_tol=mas_click_tol,
                        mas_drag_tol=mas_drag_tol,
                        mas_finished_loop_k=mas_finished_loop_k,
                    )
                    break
                except Exception as e:
                    if "Device is not healthy" in str(e) and retry_on_device_unhealthy > 0:
                        logger.warning("Device is not healthy, retrying...")
                        time.sleep(20)
                        retry_on_device_unhealthy -= 1
                        traj_logger.reset_traj()
                        continue
                    else:
                        logger.exception(f"Error executing task {task_name}")
                        return None

            task_duration = time.time() - task_start_time
            task_success = task_score > 0.0

            logger.info(
                "Task '{}' completed on {}: success={}, score={}, steps={}, duration={:.1f}s",
                task_name,
                env.base_url,
                task_success,
                task_score,
                task_steps,
                task_duration,
            )

            return {
                "task_name": task_name,
                "score": task_score,
            }
    finally:
        # Remove the thread-specific handler
        logger.remove(thread_handler_id)
        env_queue.put((env, container_name))


def _init_env(
    env_url: str, device: str, step_wait_time: float, suite_family: str, enable_mcp: bool
) -> AndroidEnvClient:
    """Initialize the environment."""
    if enable_mcp:
        env = AndroidMCPEnvClient(env_url, device, step_wait_time=step_wait_time)
    else:
        env = AndroidEnvClient(env_url, device, step_wait_time=step_wait_time)
    env.switch_suite_family(suite_family)
    return env


def run_agent_with_evaluation(
    agent_type: str,
    model_name: str,
    llm_base_url: str,
    log_file_root: str,
    tasks: list[str],
    max_step: int = -1,
    aw_urls: list[str] | None = None,
    api_key: str | None = None,
    device: str = "emulator-5554",
    step_wait_time: float = 1.0,
    suite_family: str = "mobile_world",
    env_name_prefix: str = "mobile_world_env",
    env_image: str = "mobile_world",
    dry_run: bool = False,
    enable_mcp: bool = False,
    enable_user_interaction: bool = False,
    max_concurrency: int | None = None,
    shuffle_tasks: bool = False,
    auto_retry: int = 10,
    checker_enabled: bool = False,
    checker_model_name: str | None = None,
    checker_base_url: str | None = None,
    checker_api_key: str | None = None,
    checker_dhash_threshold: int = 4,
    checker_no_change_k: int = 3,
    checker_enable_answer_trigger: bool = True,
    mas_enabled: bool = False,
    mas_skip_recent_k: int = 8,
    mas_recent_window: int = 5,
    mas_click_tol: int = 80,
    mas_drag_tol: int = 100,
    mas_finished_loop_k: int = 3,
    **kwargs,
) -> list[dict]:
    """Run the agent and return the evaluation results.

    Args:
        agent_type: Type of agent to use
        model_name: Model name for the agent
        llm_base_url: LLM service base URL
        log_file_root: Root directory for log files
        tasks: List of task names to execute (empty list for all tasks)
        max_step: Maximum steps for task execution
        aw_urls: List of Android World backend URLs. If None, auto-discover from containers
        api_key: API key for LLM service
        device: Android device ID
        step_wait_time: Wait time after each step
        suite_family: Suite family to use
        **kwargs: Additional kwargs for agent creation

    Returns:
        list[dict]: The evaluation results for each task, containing task_name, success, score, steps, duration_seconds, env_url
    """

    container_names = None
    if aw_urls is None or len(aw_urls) == 0:
        logger.info("No backend URLs specified, auto-discovering from containers...")
        aw_urls, container_names = discover_backends(image_filter=env_image, prefix=env_name_prefix)
        logger.info("Container names: {}", container_names)
        if not aw_urls:
            logger.error("No backend URLs found. Please start containers or specify --aw-host")
            return [], []

    logger.info("Using {} backend URL(s): {}", len(aw_urls), aw_urls)

    envs = Parallel(
        n_jobs=min(max_concurrency if max_concurrency is not None else len(aw_urls), len(aw_urls)),
        backend="threading",
    )(
        delayed(_init_env)(env_url, device, step_wait_time, suite_family, enable_mcp)
        for env_url in aw_urls
    )

    if len(tasks) != 0:
        task_list = tasks
    else:
        task_list = envs[0].get_suite_task_list(enable_mcp=enable_mcp, enable_user_interaction=enable_user_interaction)

    logger.info("Task list: {} ({} tasks)", task_list, len(task_list))

    num_envs = len(envs)
    max_attempts = min(1 + auto_retry, 10)  # Cap at 10 to prevent infinite loops

    for attempt in range(max_attempts):
        # Scan finished tasks each iteration (picks up results from previous attempts)
        finished_task_list, finished_scores = scan_finished_tasks(log_file_root, task_list)
        logger.info("Finished task list: {} ({} tasks)", finished_task_list, len(finished_task_list))

        pending_tasks = [task for task in task_list if task not in finished_task_list]
        logger.info(
            "Attempt {}/{}: {} remaining tasks to execute",
            attempt + 1, max_attempts, len(pending_tasks),
        )

        if not pending_tasks:
            logger.info("All tasks finished, no retry needed")
            break

        env_queue = Queue[tuple[AndroidEnvClient, str | None]](maxsize=num_envs)
        for i, env in enumerate(envs):
            env_queue.put((env, container_names[i] if container_names else None))

        if shuffle_tasks:
            random.shuffle(pending_tasks)

        if not dry_run:
            task_results = Parallel(
                n_jobs=min(max_concurrency if max_concurrency is not None else num_envs, num_envs),
                backend="threading",
            )(
                delayed(_process_task_on_env)(
                    task_name=task_name,
                    env_queue=env_queue,
                    agent_type=agent_type,
                    model_name=model_name,
                    llm_base_url=llm_base_url,
                    api_key=api_key,
                    log_file_root=log_file_root,
                    max_step=max_step,
                    enable_mcp=enable_mcp,
                    checker_enabled=checker_enabled,
                    checker_model_name=checker_model_name,
                    checker_base_url=checker_base_url,
                    checker_api_key=checker_api_key,
                    checker_dhash_threshold=checker_dhash_threshold,
                    checker_no_change_k=checker_no_change_k,
                    checker_enable_answer_trigger=checker_enable_answer_trigger,
                    mas_enabled=mas_enabled,
                    mas_skip_recent_k=mas_skip_recent_k,
                    mas_recent_window=mas_recent_window,
                    mas_click_tol=mas_click_tol,
                    mas_drag_tol=mas_drag_tol,
                    mas_finished_loop_k=mas_finished_loop_k,
                    **kwargs,
                )
                for task_name in pending_tasks
            )
        else:
            logger.info("Dry run mode, skipping task execution")
            task_results = []
            break

        # Identify failed tasks for potential retry
        failed_this_round = [
            task_name for task_name, task_result in zip(pending_tasks, task_results)
            if task_result is None
        ]

        logger.info(
            "Attempt {}/{} done: {} succeeded, {} failed/stale",
            attempt + 1, max_attempts,
            len(pending_tasks) - len(failed_this_round), len(failed_this_round),
        )

        if not failed_this_round or attempt >= max_attempts - 1:
            break

        logger.info("Auto-retrying {} failed tasks (retry {}/{})", len(failed_this_round), attempt + 1, auto_retry)

    # Final scan to get all finished results (including from retries)
    finished_task_list, finished_scores = scan_finished_tasks(log_file_root, task_list)
    # Build final results from scan (authoritative source)
    success_task_results = []
    for task_name, score in zip(finished_task_list, finished_scores):
        success_task_results.append({"task_name": task_name, "score": score})

    task_list_with_no_results = [task for task in task_list if task not in finished_task_list]
    logger.info(f"Final: {len(success_task_results)} tasks with results, {len(task_list_with_no_results)} with no results")

    return (success_task_results, task_list_with_no_results)
