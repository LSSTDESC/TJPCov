import logging, os


class LogFactory:
    @classmethod
    def Create(
        cls,
        log_nm="TJPCov",
        min_log_level=logging.DEBUG,
        fmt="'%(name)-12s: %(levelname)-8s %(message)s'",
        log_file=None,
    ) -> logging.Logger:
        logging.basicConfig(level=min_log_level, format=fmt)

        logger = logging.getLogger(log_nm)

        if log_file is not None:
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))

            fh = logging.FileHandler(log_file)
            fh.setLevel(min_log_level)
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)

        return logger
