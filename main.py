import sys
from predictor import ColorPredictor
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py data_path output_path")
        return 
    else:
        data_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
    
    clr_predictor = ColorPredictor(data_path,
                                   output_path,
                                   train_if_not_trained=True,
                                   train_regardless=False)
    df = clr_predictor.infer(save_to_file=True)
    clr_predictor.evaluate(df, True)

if __name__ == "__main__":
    main()
