import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda():
    logger.info("Testing CUDA availability...")
    if torch.cuda.is_available():
        logger.info(f"CUDA is available")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test CUDA operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        logger.info("CUDA operations test successful")
    else:
        logger.warning("CUDA is not available")

def test_data():
    logger.info("Testing data file availability...")
    data_path = os.path.join("data", "raw", "train.csv")
    if os.path.exists(data_path):
        logger.info(f"Data file found at {data_path}")
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"Dataset size: {len(df)} rows")
    else:
        logger.error(f"Data file not found at {data_path}")

if __name__ == "__main__":
    test_cuda()
    test_data() 