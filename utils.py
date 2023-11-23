import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#import ipywidgets as widgets
#%matplotlib inline
#from IPython.display import HTML
import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path
import pathlib
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import cdist
import glob
from typing import List, Dict, Any, Tuple, Union
import time
from functools import wraps
from collections import defaultdict

class ColumnNames:
    true_label = 'true_color'
    file = 'file'
    dataset = 'dataset'
    pred_occupancy_color = 'pred_occupancy_color'
    pred_hist_color = 'pred_hist_color'
    color_names =('black', 'blue', 'brown', 'green', 'pink', 'red', 'silver', 'white', 'yellow')
    histograms = 'histograms'


class IntermediateColumnNames:
    """These column names are used for storing intermediate images, useful for debugging and demonstrations"""
    original = 'original'
    cropped = 'cropped'
    sv_mask = 'sv_mask'
    edges_mask = 'edges_mask'
    focal_mask = 'focal_mask'
    mask='mask'
    total_mask='total_mask'
    hue_histogram = 'hue_histogram'
    sat_histogram = 'sat_histogram'
    val_histogram = 'val_histogram'
    val_histogram_before_equalization = 'val_histogram_before_equalization'

    @staticmethod
    def to_dict():
        return {key: value for key, value in vars(IntermediateColumnNames).items()
                if not callable(value) and not key.startswith('__') and not key.startswith('to_dict')}


def plot_hue_sat_val_histograms(hist_hue, hist_sat, hist_val, title=''):
        fig, ax = plt.subplots(1,3, figsize=(20,7))
        ax[0].plot(hist_hue)
        ax[0].set_xlabel('Hue', fontsize=16)
        ax[1].plot(hist_sat)
        ax[1].set_xlabel('Saturation', fontsize=16)
        ax[2].plot(hist_val)
        ax[2].set_xlabel('Value', fontsize=16)
        # plt.plot(hist_hue, color='g')
        fig.suptitle(title, fontsize=25) 
        #plt.xlabel('Pixel Intensity')
        #plt.ylabel('Frequency')
        plt.show()

# def showcase_image(original,
#                     cropped, 
#                     sv_mask, 
#                     edges_mask,
#                     mask, total_mask, 
#                     hist_hue,
#                     hist_sat,
#                     hist_val,

#                     ):
#     pass

def plot_images(n_columns, **kwargs):
    n_images = len(kwargs)
    # n_columns = 4
    n_rows = int(np.ceil(n_images / n_columns))

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 10, n_rows * 10))  # Adjust figsize
    
    for i, (key, image_data) in enumerate(kwargs.items()):
        ax = axs.flatten()[i]  # Access the current axis in a flattened array
        ax.imshow(image_data) 
        ax.set_title(key, fontsize=50)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

def print_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for '{func.__name__}': {execution_time:.6f} seconds")
        return result
    return wrapper

def mlib_rgb_to_cv2_rgb(img):
    return (img * 255).astype(np.uint8)

def imshow(img, title='', show=True, path=''):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    if show:
        plt.show()
    if path != '':
        fig.savefig(path)
    plt.close(fig)

# Define a function to plot the selected color as a rectangle
def plot_rgb(rgb):
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Plot a rectangle with the color
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=rgb))
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Show the plot
    plt.show()

def mean_color(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        img (np.ndarray): _description_
        mask (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    if len(mask.shape) == 2:
        return np.sum(img * np.expand_dims(mask, -1), axis=(0,1) ) / np.sum(mask)
    if len(mask.shape) == 3:
        return np.sum(img *  mask) / np.sum(mask)
    
# ================================================== MASKS =========================================================================================
# pixels with a value of 1 in a mask or kept, while values of 0 are discarded (in ColorClassifier class)

def focal_spread_mask(width: int, height: int, focal_x: int=None, focal_y:int=None, min_val=0.1) -> np.ndarray:
    """_summary_

    Args:
        width (int): _description_
        height (int): _description_
        focal_x (_type_, optional): _description_. Defaults to None.
        focal_y (_type_, optional): _description_. Defaults to None.
        min_val (float, optional): _description_. Defaults to 0.5.

    Returns:
        np.ndarray: _description_
    """

    if focal_x is None:
        focal_x = width//2
    if focal_y is None:
        focal_y = height//2
        

    # Create a meshgrid of pixel coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Stack the X and Y coordinates into a 2D array
    coords = np.stack((X, Y), axis=-1)
    
    #print(coords.shape)

    # Create a 2D array with the focal pixel coordinate
    focal = np.array([[focal_x, focal_y]])

    # Calculate the distance matrix between the pixel coordinates and the focal pixel
    dmat = cdist(coords.reshape(-1, 2), focal).reshape(height, width)
    
    #print(dmat.shape)

    # Find the minimum and maximum distance values
    dmin = dmat.min()
    dmax = dmat.max()

    # Scale the distance matrix linearly from 1 to min_val
    scaled_dmat = 1 - (1- min_val) * (dmat - dmin) / (dmax - dmin)

    return scaled_dmat # 224, 224

 # ------------------------------------------------------------------------------------------------------   
def hsv_percentile_mask(hsv_img : np.ndarray, s_min=None, s_max=None, v_min=None, v_max=None ) -> np.ndarray:
    """_summary_

    Args:
        hsv (np.ndarray): HSV image
        s_min (float, optional): min saturation percentile. Defaults to None.
        s_max (float, optional): max saturation percentile. Defaults to None.
        v_min (float, optional): min value percentile. Defaults to None.
        v_max (float, optional): max value percentile. Defaults to None.

    Returns:
        np.ndarray: binary mask of same shape
    """

    mask = np.zeros(shape=hsv_img.shape[0:2], dtype=bool)
    v = hsv_img[:,:,2]
    s = hsv_img[:,:,1]
    if s_min is not None:
        mask = np.bitwise_or(mask, s < np.percentile(s, s_min))
    if s_max is not None:
        mask = np.bitwise_or(mask, s > np.percentile(s, s_max))
        
    if v_min is not None:
        mask = np.bitwise_or(mask, v < np.percentile(v, v_min))
    if v_max is not None:
        mask = np.bitwise_or(mask, v > np.percentile(v, v_max))
        
    return ~mask

# ------------------------------------------------------------------------------------------------------
def hsv_inrange_mask(hsv_img: np.ndarray, lower_hsv : float, upper_hsv: float) -> np.ndarray:
    """Creates a mask of the HSV color

    Args:
        hsv_img (np.ndarray): HSV image
        lower_hsv (float): lower bounds of HSV 
        upper_hsv (float): upper bounds of HSV

    Returns:
        np.ndarray: _description_
    """

    if lower_hsv[0] > upper_hsv[0]:
        mask = np.logical_or(hsv_img[:,:,0] >= lower_hsv[0], hsv_img[:,:,0] <= upper_hsv[0])
    else:
        mask = np.logical_and(hsv_img[:,:,0] >= lower_hsv[0], hsv_img[:,:,0] <= upper_hsv[0])
    mask = np.logical_and(mask, hsv_img[:,:,1] >= lower_hsv[1])
    mask = np.logical_and(mask, hsv_img[:,:,1] <= upper_hsv[1])
    mask = np.logical_and(mask, hsv_img[:,:,2] >= lower_hsv[2])
    mask = np.logical_and(mask, hsv_img[:,:,2] <= upper_hsv[2])

    return mask

# ------------------------------------------------------------------------------------------------------
def get_edges_mask(img: np.ndarray, val: float = 200, blur_ksize: float=3) -> np.ndarray:
    """Creates a mask such that edges are ignored, since they present transitions and therefore uncertainty
    and usually some shading and reflections. Furthermore, car wheels have a lot of edges and needs to ignored.
    The edges are blurred to enlarge the mask

    Args:
        img (np.ndarray): np.uint8 for cv2 compatibility
        val (float, optional): val for canny algorithm. Defaults to 200.
        blur_ksize (float, optional): bluring kernel size. Defaults to 3.

    Returns:
        np.ndarray: Mask of where there are NOT edges
    """
    edges = cv2.Canny(img, val, val+20)
    edges_blurred = cv2.blur(edges,(blur_ksize,blur_ksize))
    mask = edges_blurred == 0.0
    return mask 

# ------------------------------------------------------------------------------------------------------
def crop_image(img: np.ndarray, ratio: float) -> np.ndarray:
    """Crops image

     Args:
         img (np.ndarray): image
         ratio (float): in range of 0 to 1

     Returns:
         np.ndarray: Cropped image
    """
    height, width = img.shape[:2]
   
    crop_top = int(height * ratio)
    crop_bottom = int(height * (1 - ratio))
    crop_left = int(width * ratio)
    crop_right = int(width * (1 - ratio))
   
    cropped_image = img[crop_top:crop_bottom, crop_left:crop_right, ...]
   
    return cropped_image  

# ================================================== DATA =========================================================================================
 
# ------------------------------------------------------------------------------------------------------
def get_files_dict_list(path: pathlib.Path,
                color_names: Tuple[str, ...] =('black', 'blue', 'brown', 'green', 'pink', 'red', 'silver', 'white', 'yellow') ) -> List[Dict]:
    """Returns a list of dictionaries, with keys for the file, true_label, and dataset (train/test/None)
    Any images at the path that are not in folders i.e. path/img.png, will not have a true label or a dataset.
    This is useful for when infering images in the ColorPredictor class. 

     Args:
         path (str | pathlib.Path): Path of folder to read from.
         color_names (Tuple[str, ...], optional): _description_. Defaults to ('black', 'blue', 'brown', 'green', 'pink', 'red', 'silver', 'white', 'yellow').

     Returns:
         List[Dict]: An element for each image
    """
    if isinstance(path, str):
        path = Path(path)
    train_path = path / 'train' 
    test_path = path / 'test'

    list_of_dict = []

    # Train set with true colors
    if train_path.exists():
        for color_name in color_names:
            color_path = train_path / color_name 
            if not color_path.exists():
                continue
            files = color_path.glob('*.jpg')
            list_of_dict += [{ColumnNames.file: f, ColumnNames.true_label: color_name, ColumnNames.dataset: 'train'} for f in files]

    # Test set with true colors
    if test_path.exists():
        for color_name in color_names:
            color_path = test_path / color_name 
            if not color_path.exists():
                continue
            files = color_path.glob('*.jpg')
            list_of_dict += [{ColumnNames.file: f, ColumnNames.true_label: color_name, ColumnNames.dataset: 'test'} for f in files]

    # unknown set with true colors
    for color_name in color_names:
        color_path = path / color_name 
        if not color_path.exists():
            continue
        print(len([f for f in files]))
        list_of_dict += [{ColumnNames.file: f, ColumnNames.true_label: color_name, ColumnNames.dataset: 'None'} for f in files]

    # unknown set with unknown colors (inference mode)
    files = path.glob('*.jpg')
    list_of_dict += [{ColumnNames.file: f, ColumnNames.true_label: 'None', ColumnNames.dataset: 'None'} for f in files]


    # return pd.DataFrame(list_of_dict)
    return list_of_dict


# ------------------------------------------------------------------------------------------------------
def filter_data(input_data: Union[List[Dict], pd.DataFrame], filter_params: Dict):
    """
    Filter input data (dictionary or DataFrame) based on specified parameters.

    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Input data to be filtered.

    filter_params : dict
        Dictionary containing filter parameters.
        Example: {'Age': 30, 'City': 'New York'}

    Returns:
    --------
    dict or pd.DataFrame
        Filtered data based on the provided parameters.
    """
    if isinstance(input_data, list):
        # Filter a dictionary
        # filtered_dict = {key: value for key, value in input_data.items() if all(value.get(k) == v for k, v in filter_params.items())}
        filtered_list = [d for d in input_data if np.all([d[key] == filter_params[key] for key in filter_params.keys()])]
        return filtered_list
    elif isinstance(input_data, pd.DataFrame):
        # Filter a DataFrame
        filter_condition = pd.Series([True] * len(input_data))
        for key, value in filter_params.items():
            filter_condition &= (input_data[key] == value)
        filtered_df = input_data[filter_condition]
        return filtered_df
    else:
        raise ValueError("Unsupported data type. Function supports dict or pd.DataFrame inputs.")
    
    
# ------------------------------------------------------------------------------------------------------
def select_top_by_class(input_data: Union[dict, pd.DataFrame], class_column: str, count: int=10) -> Union[Dict,  pd.DataFrame]:
    """Selects the top 10 rows for each class. Useful for quick debugging and demonstrations. 

    Args:
        input_data (Union[dict, pd.DataFrame])
        class_column (str): _description_
        count (int, optional):  Defaults to 10.

    Returns:
        Union[Dict,  pd.DataFrame]: The smaller data structire of type as input_data
    """
    if isinstance(input_data, pd.DataFrame):
        class_values = input_data[class_column].unique()
        selected_rows = pd.DataFrame()
        for value in class_values:
            class_subset = input_data[input_data[class_column] == value].head(count)
            selected_rows = pd.concat([selected_rows, class_subset])
        return selected_rows
    elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        class_dict = defaultdict(list)
        for item in input_data:
            class_dict[item[class_column]].append(item)
        
        selected_rows = []
        for _, value in class_dict.items():
            selected_rows.extend(value[:count])
        return selected_rows
    else:
        raise ValueError("Unsupported data type. Function supports list of dictionaries or pd.DataFrame inputs.")


# ================================================== HISTOGRAM DISTANCES =========================================================================================

# ------------------------------------------------------------------------------------------------------
def histogram_distances(images_histograms: np.ndarray, color_histograms_means: Dict, color_histograms_stds: Dict, hsv_color_weights: Dict) -> Dict:
    """For every image, determines the distance to each color by comparing the histograms. 
    The average square distance between each bin is normalized by the color std values.

    dist for color = sqrt( sum(  ((b_color - b_img)/ b_color_std )**2 )),
    where the sum is computed over all (b)ins, which usually is 256*3 = 768 bins


    Args:
        hist_hsv_bulk (np.ndarray): Shape = n_imgs, 3, bins, i.e. the histograms for each image
        color_histograms_means (Dict): Each color has an array of shape (3, bins)
        color_histograms_stds (Dict): Each color has an array of shape (3, bins)
        hsv_color_weights (Dict): Each color has an array of shape (3,), which weights the importance of the H, S and V, e.g. 

    Returns:
        Dict: each color key has a list with length equal to the number of images
    """
    distances_dict = {}
    #print(hist_hsv_bulk.shape)
    for color in list(color_histograms_means.keys()):
        mean = color_histograms_means[color] # shape = (3, bins )
        std = color_histograms_stds[color] # shape = (3, bins )
        hsv_weights = np.expand_dims(np.array(hsv_color_weights[color]), 1) # shape = (3,1)
        # print(std.mean())
        
        # dist = np.linalg.norm(hist_hsv_bulk - mean, axis=(2)) # shape - (395,3)
        # dist = np.dot(dist, np.array(hsv_weights) )
        
        std = (std / hsv_weights) + 0.001 # stability
        squared_diff = ((images_histograms - mean) / std) **2 # 395,3, bins
        squared_sum = np.sum(squared_diff, axis=(1,2)) # shape = n_images
        dist = np.sqrt(squared_sum) # shape = n_images

        distances_dict[f'{color}_dist'] = dist
    return distances_dict

def distances_dict_to_df(distances_dict: Dict):
    return pd.DataFrame.from_dict(distances_dict, orient='index').transpose()

def smallest_distance_label(distances: Union[Dict, pd.DataFrame]) -> List[str]:
    """For each row: Determines the column that has the smallest value and extracts the label

    Args:
        distances (Union[Dict, pd.DataFrame]): rows: observations, columns: color_distances AND NO OTHER COLUMNS

    Returns:
        List: list of labels
    """

    if isinstance(distances, dict):
        distances_df = distances_dict_to_df(distances)
        # distances_df = pd.DataFrame(distances)
        min_keys = distances_df.idxmin(axis=1) # assumes there are no other keys, a bit risk
        return [v.split('_')[0] for v in min_keys.values]
    elif isinstance(distances, pd.DataFrame):
        min_keys = distances.idxmin(axis=1) # assumes there are no other columns, risky
        return [v.split('_')[0] for v in min_keys.values]
    else:
        raise ValueError("Unsupported data type. Function supports dict or pd.DataFrame inputs.")

# ------------------------------------------------------------------------------------------------------
def largest_occupancy_label(df) -> List[str]:
    """Determines column with largest value and extracts the label from the column

    Args:
        df (_type_): dataframe with color_occupancy columns

    Returns:
        List[str]: list of colors, length equal to number of rows 
    """
    # List of columns containing float values for consideration
    occupancy_columns = [f'{col}_occupancy' for col in ColumnNames.color_names ]

    # Function to get the column name with the highest value for each row
    def get_max_column(row):
        max_column = np.argmax(row[occupancy_columns])
        return occupancy_columns[max_column].split('_')[0]

    # Apply the function row-wise and store the result in a new column
    # df['pred_pixel_label'] = df.apply(get_max_column, axis=1)
    labels = df.apply(get_max_column, axis=1)
    return labels
    
# ================================================== EVLAUATION  =========================================================================================

# ------------------------------------------------------------------------------------------------------
def confusion_matrix_df(df: pd.DataFrame, true_label_column='true_color', pred_label_column: str='pred_color') -> pd.DataFrame:
    """Calculate confusion matrix and returns df

    Args:
        df (pd.DataFrame): Needs at least 2 columns
        true_label_column (str, optional):  Defaults to 'true_color'.
        pred_label_column (str, optional):  Defaults to 'pred_color'.

    Returns:
        pd.DataFrame: confusion matrix
    """
    
    confusion_matrix = pd.crosstab(df[true_label_column], df[pred_label_column], rownames=['True'], colnames=['Predicted'])

    total_predicted = confusion_matrix.sum(axis=0)
    total_true = confusion_matrix.sum(axis=1)

    # Create DataFrame for total counts
    total_predicted_df = pd.DataFrame([total_predicted.values], columns=total_predicted.index, index=['Total Predicted'])
    total_true_df = pd.DataFrame(total_true, columns=['Total True'])

    # Concatenate the original confusion matrix and the total counts
    confusion_matrix_with_totals = pd.concat([confusion_matrix, total_true_df], axis=1)
    confusion_matrix_with_totals = pd.concat([confusion_matrix_with_totals, total_predicted_df])
    
    return confusion_matrix_with_totals

# ------------------------------------------------------------------------------------------------------
def create_evaluation_df(df, true_label_column='true_color', pred_label_column='pred_color') -> pd.DataFrame:
    """Calculate F1, recall, precision for each class and overall; returns a df of values

    Args:
        df (_type_): Needs at least 2 columns
        true_label_column (str, optional): . Defaults to 'true_color'.
        pred_label_column (str, optional): . Defaults to 'pred_color'.

    Returns:
        pd.DataFrame: recall, precision, and F1 for each class and overall 
    """
    # pred_label_column_name = 'pred_pixel_label'

    # Get the true and predicted labels from the DataFrame
    true_labels = df[true_label_column]
    predicted_labels = df[pred_label_column]

    # Compute the confusion matrix using sklearn
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Get all unique labels (colors) from the DataFrame
    unique_labels = df[true_label_column].unique()

    # Compute metrics for each color
    metrics = {}
    for label in unique_labels:
        # Get the index of the label in the unique labels list
        label_index = list(unique_labels).index(label)

        # Calculate precision, recall, and F1-score for each label
        precision = precision_score(true_labels, predicted_labels, labels=[label], average='micro')
        recall = recall_score(true_labels, predicted_labels, labels=[label], average='micro')
        f1 = f1_score(true_labels, predicted_labels, labels=[label], average='micro')

        # Store metrics for each label in a dictionary
        metrics[label] = {'Precision': precision, 'Recall': recall, 'F1-score': f1}

    # Compute overall metrics
    overall_precision = precision_score(true_labels, predicted_labels, average='micro')
    overall_recall = recall_score(true_labels, predicted_labels, average='micro')
    overall_f1 = f1_score(true_labels, predicted_labels, average='micro')
    overall_metrics = {'Precision': overall_precision, 'Recall': overall_recall, 'F1-score': overall_f1}


    # Convert class-wise metrics to DataFrame
    class_df = pd.DataFrame(metrics).T

    # Convert overall metrics to DataFrame
    overall_df = pd.DataFrame(overall_metrics, index=['Overall']).T

    # Concatenate both DataFrames
    final_df = pd.concat([class_df, overall_df], axis=1)
    #final_df = class_df.append(overall_df.T)

    return final_df


