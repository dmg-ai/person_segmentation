from config import AppConfig

from clearml.automation import TaskScheduler


def run_scheduler(config):
    scheduler = TaskScheduler()
    scheduler.add_task(
        schedule_task_id=config.scheduler_task_id,
        queue=config.scheduler_queue,
        minute=config.scheduler_minute,
        hour=config.scheduler_hour,
        day=config.scheduler_day,
        weekdays=config.scheduler_weekdays,
        target_project=config.project_name,
    )
    scheduler.start_remotely(queue=config.scheduler_queue)


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    run_scheduler(config)
