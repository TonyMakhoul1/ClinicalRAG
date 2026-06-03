import logging
import os
import sys

import structlog


def setup_logging() -> None:
    log_format = os.getenv("LOG_FORMAT", "console")

    # These run on EVERY log event
    shared_processors = [
        # adds the bind to every event, this is how request_id propagates automatically to every log
        structlog.contextvars.merge_contextvars,

        # tell which module logged this event (retrieval, app,...)
        structlog.stdlib.add_logger_name,

        # Add the level as a string: "info", "warning", "error", etc.
        # which type (info, warning, error...)
        structlog.stdlib.add_log_level,

        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),

        # Render stack_info= field if it was passed (rare, but correct to handle).
        # If the field was passed...
        structlog.processors.StackInfoRenderer(),

        # get a readable traceback string, wihtout this raw python tuple in JSON.
        structlog.processors.format_exc_info,
    ]

    # LOG_FORMAT=consol for local dev, json for docker/production(adds LOG_FORMAT=json in docker-compose.yml)
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        # take the exicting processor
        # the fct means make structlog compatible with normal logging
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],

        # tell structlog to create loggers using built-in logging module
        logger_factory=structlog.stdlib.LoggerFactory(),
        # the type of of logger (can do logger.bind)
        wrapper_class=structlog.stdlib.BoundLogger,
        # save the logger after creating it the first time
        cache_logger_on_first_use=True,
    )

    # create a formatter for logs(controls how logs look when printed)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            # remove internal structlog, cz it adds extra technical info
            # render decide how the final log appears(json,consol)
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        # let the logs coming from Python logging
        foreign_pre_chain=shared_processors,
    )
    # handler to print logs
    handler = logging.StreamHandler(sys.stdout)
    # every log printed uses the formatting rules
    handler.setFormatter(formatter)

    # get the root logger
    root = logging.getLogger()
    root.handlers.clear()   # remove any handlers added before setup_logging() ran
    # add the custom handler to the root logger
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # These reduce unnecessary logs from libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("crewai").setLevel(logging.WARNING)
