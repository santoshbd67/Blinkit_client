#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:26:44 2023

@author: Parmesh
"""
import logging
import logging.config
# import yaml
import os
import TAPPconfig as cfg
from typing import Dict, Any
import json


# def load_logging_configuration() -> Dict[str, Any]:
#     """
#     Load the logging configuration from a YAML file.

#     Args:
#         config_file (str): Path to the logging configuration file.

#     Returns:
#         dict: Loaded logging configuration.

#     Raises:
#         FileNotFoundError: If the specified config file is not found.
#         yaml.YAMLError: If there is an error parsing the YAML configuration file.
#     """
#     try:
#         config_file = cfg.get_logging_config()
#         with open(config_file, 'r') as f:
#             config = yaml.safe_load(f)
#     except FileNotFoundError as ex:
#         print(f"Error: Failed to load logging configuration from {config_file}")
#         raise ex
#     except yaml.YAMLError as ex:
#         print(f"Error: Failed to parse YAML configuration file: {config_file}")
#         raise ex
#     try:
#         # Override configuration values with environment variables
#         if cfg.get_loggin_level():
#             config["handlers"]["file_handler"]['level'] = cfg.get_loggin_level() #os.environ['LOG_LEVEL']
#         if cfg.get_logging_format():
#             config['formatters']['simple']['format'] = cfg.get_logging_format() #os.environ['LOG_FORMAT']
#     except:
#         pass
#     return config


def load_env_as_json() -> Dict[str, Any]:
    """
    Load the .env file as a JSON object.

    Returns:
        dict: JSON object representing the .env file contents.
    """
    try:
        with open(".env", "r") as env_file:
            json_object = json.load(env_file)

        return json_object
    except Exception as ex:
        print("Exp occured while reading .env",ex)
        return {}

def load_logging_configuration(new_logger : str = None) -> Dict[str, Any]:
    """
    Load the logging configuration from a JSON file and override values with environment variables.

    Returns:
        dict: Loaded logging configuration.

    Raises:
        FileNotFoundError: If the specified config file is not found.
        json.JSONDecodeError: If there is an error parsing the JSON configuration file.
    """
    try:
        config_file = "./config.json"  # cfg.get_logging_config()
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError as ex:
        print(f"Error: Failed to load logging configuration from {config_file}")
        raise ex
    except json.JSONDecodeError as ex:
        print(f"Error: Failed to parse JSON configuration file: {config_file}")
        raise ex
    config = config["logging_config"]
    try:
        # Load .env file as a JSON object
        env_json = load_env_as_json()

        # Override configuration values with environment variables from the JSON object
        logger_name = env_json.get("LOGGER_NAME", config["logger_name"])
        console_level = env_json.get("CONSOLE_LEVEL", config["handlers"]["console"]["level"])
        file_handler_level = env_json.get("FILE_HANDLER_LEVEL", config["handlers"]["file_handler"]["level"])
        logging_format = env_json.get("LOGGING_FORMAT", config["formatters"]["standard"]["format"])

        # Update logger name if specified in environment variable
        if new_logger:
            print("new_logger :",new_logger)
            logger_name = new_logger
        config["logger_name"] = logger_name

        # Update logger name if specified in environment variable
        config["loggers"][logger_name] = config["loggers"].pop("my_logger")
        print("logger name :",config["logger_name"], "config :",config.get("loggers").get(logger_name))
        # Update handler levels based on environment variables
        config["handlers"]["console"]["level"] = console_level
        config["handlers"]["file_handler"]["level"] = file_handler_level

        # Update logging format based on environment variable
        config["formatters"]["standard"]["format"] = logging_format

        return config
    except Exception as e:
        print("Exception occurred", e)
        return config


def setup_logging(log_file: str, logger_name: str = None) -> logging.Logger:
    """
    Setup logging configuration and return a logger object.

    Args:
        log_file (str): Path to the log file.
        logger_name (str): Name of the logger. If None, use the default logger name from the configuration.

    Returns:
        logging.Logger: Configured logger object.

    """
    config = load_logging_configuration(new_logger=logger_name)
    config['handlers']['file_handler']['filename'] = log_file

    logging.config.dictConfig(config)
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger(config.get("logger_name", "default_logger"))

    # Add a method to close the file handler
    def close_file_handler(file_path: str = log_file):
        """
        Close the file handler for the specified log file and remove it from the logger.

        Args:
            file_path (str): Path to the log file.
        """
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_path:
                handler.close()
                handlers_to_remove.append(handler)
        for handler in handlers_to_remove:
            logger.removeHandler(handler)

    logger.close_file_handler = close_file_handler

    return logger


def get_logger_old(logger_name:str= None) -> logging.Logger:
    """
    Get the logger object.

    Returns:
        logging.Logger: Logger object.

    """
    if logger_name:
        conf_ = load_logging_configuration(new_logger = logger_name)
    else:
        conf_ = load_logging_configuration()
    logger_name = conf_.get("logger_name")
    if logging.getLogger(logger_name).hasHandlers():
        return logging.getLogger(logger_name)
    else:
        logger = logging.getLogger(logger_name)

        if not logger.hasHandlers():
            # config = load_logging_configuration()
            logging.config.dictConfig(conf_)

        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        return logger

def get_logger(logger_name: str = None, log_file: str = None) -> logging.Logger:
    """
    Get the logger object.
    Args:
        logger_name (str): Name of the logger.
        log_file (str): Path to the log file.
    Returns:
        logging.Logger: Logger object.
    """
    if logger_name and log_file:

        # Check if the logger already exists with the given name
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            # If the logger does not have any handlers, it does not exist.
            # Set up the logging configuration and create a new logger.
            logger = setup_logging(log_file, logger_name=logger_name)
        else:
            # The logger already exists. Check if it has the specified log file handler.
            matching_handler = next((handler for handler in logger.handlers if hasattr(handler, 'baseFilename') and handler.baseFilename == log_file), None)
            if not matching_handler:
                # If the logger exists but does not have the specified log file handler, create a new handler and add it to the logger.
                new_handler = logging.FileHandler(log_file)
                new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(new_handler)
        print("returning existing logger with hadler")
        return logger
    elif logger_name and log_file is None:
        conf_ = load_logging_configuration(new_logger = logger_name)
    else:
        conf_ = load_logging_configuration()
    logger_name = conf_.get("logger_name")
    if logging.getLogger(logger_name).hasHandlers():
        return logging.getLogger(logger_name)
    else:
        logger = logging.getLogger(logger_name)
        if not logger.hasHandlers():
            logging.config.dictConfig(conf_)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
    return logger
if __name__ == "__main__":

    # z = setup_logging("test.log")
    # z.info("test log")
    # z.warning("testing started")
    # z.error("this is error")
    # z.debug("this is debug")
    y = setup_logging("y.log",logger_name="y")
    print("y logger",y)
    y.info("test y log with y.log")
    x = get_logger(logger_name="y",log_file="y.log")
    print("get new logger obj :",x)
    x.info("test x obej of y log with y.log")
    # z.close_file_handler()
    # y.close_file_handler()
    a = get_logger(logger_name="y",log_file="a.log")
    print("get new logger obj a :",a)
    a.info("Test a of y, without log file")