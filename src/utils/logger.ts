// src/utils/logger.ts
// Structured logging utility for LegalMind Backend

type LogLevel = "debug" | "info" | "warn" | "error";

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  service: string;
  traceId?: string;
  userId?: string;
  duration?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Get log level from environment
 */
function getLogLevel(): LogLevel {
  const level = process.env.LOG_LEVEL?.toLowerCase() as LogLevel;
  if (["debug", "info", "warn", "error"].includes(level)) {
    return level;
  }
  return process.env.NODE_ENV === "production" ? "info" : "debug";
}

/**
 * Check if a log level should be logged
 */
function shouldLog(level: LogLevel): boolean {
  const levels: LogLevel[] = ["debug", "info", "warn", "error"];
  const currentLevel = getLogLevel();
  return levels.indexOf(level) >= levels.indexOf(currentLevel);
}

/**
 * Format log entry for output
 */
function formatLog(entry: LogEntry): string {
  return JSON.stringify(entry);
}

/**
 * Create a log entry
 */
function createLog(
  level: LogLevel,
  message: string,
  context?: {
    traceId?: string;
    userId?: string;
    duration?: number;
    metadata?: Record<string, unknown>;
  },
): LogEntry {
  return {
    timestamp: new Date().toISOString(),
    level,
    message,
    service: "legal-mind-backend",
    ...context,
  };
}

/**
 * Logger instance with structured logging
 */
export const logger = {
  debug(
    message: string,
    context?: {
      traceId?: string;
      userId?: string;
      duration?: number;
      metadata?: Record<string, unknown>;
    },
  ): void {
    if (shouldLog("debug")) {
      console.debug(formatLog(createLog("debug", message, context)));
    }
  },

  info(
    message: string,
    context?: {
      traceId?: string;
      userId?: string;
      duration?: number;
      metadata?: Record<string, unknown>;
    },
  ): void {
    if (shouldLog("info")) {
      console.info(formatLog(createLog("info", message, context)));
    }
  },

  warn(
    message: string,
    context?: {
      traceId?: string;
      userId?: string;
      duration?: number;
      metadata?: Record<string, unknown>;
    },
  ): void {
    if (shouldLog("warn")) {
      console.warn(formatLog(createLog("warn", message, context)));
    }
  },

  error(
    message: string,
    context?: {
      traceId?: string;
      userId?: string;
      duration?: number;
      metadata?: Record<string, unknown>;
      error?: Error;
    },
  ): void {
    if (shouldLog("error")) {
      const errorContext = context?.error
        ? {
            ...context,
            metadata: {
              ...context?.metadata,
              errorMessage: context.error.message,
              errorStack: context.error.stack,
            },
          }
        : context;
      console.error(formatLog(createLog("error", message, errorContext)));
    }
  },

  /**
   * Log API request
   */
  request(
    method: string,
    path: string,
    context?: {
      traceId?: string;
      userId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void {
    this.info(`API Request: ${method} ${path}`, context);
  },

  /**
   * Log API response
   */
  response(
    method: string,
    path: string,
    statusCode: number,
    duration: number,
    context?: {
      traceId?: string;
      userId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void {
    this.info(`API Response: ${method} ${path} - ${statusCode}`, {
      ...context,
      duration,
    });
  },

  /**
   * Log analysis event
   */
  analysis(
    event: string,
    context?: {
      traceId?: string;
      userId?: string;
      duration?: number;
      metadata?: Record<string, unknown>;
    },
  ): void {
    this.info(`Analysis: ${event}`, context);
  },
};

/**
 * Create a child logger with persistent context
 */
export function createLogger(context: {
  traceId?: string;
  userId?: string;
}): typeof logger {
  return {
    ...logger,
    debug: (message: string, additionalContext?: Record<string, unknown>) =>
      logger.debug(message, { ...context, ...additionalContext }),
    info: (message: string, additionalContext?: Record<string, unknown>) =>
      logger.info(message, { ...context, ...additionalContext }),
    warn: (message: string, additionalContext?: Record<string, unknown>) =>
      logger.warn(message, { ...context, ...additionalContext }),
    error: (message: string, additionalContext?: Record<string, unknown>) =>
      logger.error(message, { ...context, ...additionalContext }),
    request: (
      method: string,
      path: string,
      additionalContext?: Record<string, unknown>,
    ) => logger.request(method, path, { ...context, ...additionalContext }),
    response: (
      method: string,
      path: string,
      statusCode: number,
      duration: number,
      additionalContext?: Record<string, unknown>,
    ) =>
      logger.response(method, path, statusCode, duration, {
        ...context,
        ...additionalContext,
      }),
    analysis: (event: string, additionalContext?: Record<string, unknown>) =>
      logger.analysis(event, { ...context, ...additionalContext }),
  };
}
