import pickle
from utils import (imshow, hsv_inrange_mask, crop_image, get_edges_mask, get_files_dict_list, filter_data, select_top_by_class,
                    histogram_distances, smallest_distance_label, confusion_matrix_df, create_evaluation_df, largest_occupancy_label, 
                    focal_spread_mask, mlib_rgb_to_cv2_rgb, hsv_percentile_mask, distances_dict_to_df, print_execution_time,
                    ColumnNames, IntermediateColumnNames)

from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import pathlib
from copy import deepcopy


class ColorPredictor:
    """
    This class predicts the overall color of objects, in this case, cars.
    There are two main approaches.
    The first finds the color with the most occupancy by looking at the HSV range.
    This is a fully hard-coded unsupervised method.

    The second method uses the H, S and V histograms of an image.
    It requires knowledge of the average and standard deviations of the histograms for each color.
    It is therefore a supervised method that requires training.
    The number of saved values that are stored to a file are : n_colors x 3 x n_bins x 2
    
    When predicting labels for each image, a label is produced for each method.
    Both methods are therefore evaluated and are completely independant.
    This comes with the downside of additional execution time in the case that we are only
    interested in one method.

    One feature per method per color is extracted, totaling 18 features per image ( we have 9 colors).
    Calculating the features takes up most of the execution time.
    Classifying an image is as simple as selecting the color with the highest value.
    Parts of the image are ignored, such as edges, before the features are calculated. 

    The HSV range method gets an overall F1 score of 74% for both test and train set.
    The histogram  method gets an overall F1 score of 80% for the train set and 72% for the test set.

    These features can also be fed into other ML classifiers.
    Most notably, we can easily get 100% accuracy with a RFC. 


    """
    # ==================================================  STATIC VARIABLES  ==================================================
    # These ranges work for matplotlib images, not for cv2
    # These values were found by looking at various histograms and by fiddling around
    # Can definitely be improved 
    hsv_ranges = {
       'black': [[0, 0, 0], [255, 255, 50]], 
       'blue': [[140, 50, 50], [180, 255, 255]],
       'brown': [[10, 25, 25], [30, 255, 255]],
       'green': [[55, 25, 25], [140, 255, 255]],
       'pink': [[180, 25, 50], [245, 255, 255]], 
       'red': [[245, 50, 50], [10, 255, 255]],
       'silver': [[110, 0, 75], [180, 100, 220]], 
       'white': [[0, 0, 200], [255, 50, 255]],
       'yellow': [[30, 50, 50], [55, 255, 255]]}
    for key in hsv_ranges.keys():
        hsv_ranges[key] = np.array(list(hsv_ranges[key])) / 255.0
    
    color_names = list(hsv_ranges.keys())

    # The H, S, and V histograms can have weights associated e.g. the black color cares very little about Hue
    # These can be found with a random grid search or through any other method
    hsv_color_weights = {'black': np.array([1.2, 0.8, 1.2]),
                        'blue': np.array([0.8, 0.9, 1. ]),
                        'brown': np.array([1. , 1.1, 1.1]),
                        'green': np.array([0.8, 0.9, 1. ]),
                        'pink': np.array([1.1, 1. , 1. ]), 
                        'red': np.array([1. , 0.9, 0.9]),
                        'silver': np.array([1. , 1.2, 0.9]),
                        'white': np.array([0.8, 0.9, 1.2]),
                        'yellow': np.array([0.8, 1.1, 1. ])}

    # # Defaults to equal weights for all
    # hsv_color_weights = {}
    # for color in color_names:
    #     hsv_color_weights[color] = (1,1,1)

    # Parameters used by the functions utils
    # We can do a random grid search to find better values, but these work well enough
    focal_spread_min_val = 0.5
    sat_percentile_min = 8
    val_percentile_min = 8
    edges_val = 160
    edges_blur_ksize = 3
    crop_ratio = 0.07
    # data_path = './dataset'
    bins = 256

    # ==================================================  CONSTRUCTOR ==================================================

    def __init__(self,
                 data_path: pathlib.Path =Path('./dataset'),
                 output_path: pathlib.Path=Path('./'),
                 hsv_hist_means_stds_path: pathlib.Path =Path('./hsv_hist_means_stds.pkl'),
                 train_if_not_trained: bool =True,
                 train_regardless: bool=False,
                 top: int=0,
                 save_intermediate_images=False
                 ) -> None:
        """
        Loads a list of dictionaries containing file paths, and true labels if available.
        Checks whether the average histograms of each color should be calculated for inference (trained).
        Processes each file by extracting some features for each image ( The bulk of execution time). 


        Args:
            data_path (pathlib.Path, optional): Datapath. Defaults to Path('./dataset').
            output_path (pathlib.Path, optional): Output path. Defaults to Path('./').
            hsv_hist_means_stds_path (pathlib.Path, optional): Path to learned/trained averages. Defaults to Path('./hsv_hist_means_stds.pkl').
            train_if_not_trained (bool, optional): Should perform training if learned average not available?. Defaults to True.
            train_regardless (bool, optional): Should perform training regardless? (retrain). Defaults to False.
            top (int, optional): Selects the top rows of each class, nice for debugging. Defaults to 0.
        """

        self.data_path = data_path
        self.output_path = output_path
        self.hsv_hist_means_stds_path = hsv_hist_means_stds_path

        self.hsv_hist_means_stds = None
        self.load_hsv_hist_means_stds()

        # List of dictionaries
        self.files_dict_list = get_files_dict_list(data_path)
        if top > 0:
            self.files_dict_list = select_top_by_class(self.files_dict_list, ColumnNames.true_label, top)


        self.trainable = False
        self.trained = False
        self.training_queued = False
        self.set_train_state(train_if_not_trained, train_regardless)

        # Calculate features
        self.save_intermediate_images = save_intermediate_images
        self.processed_files_dict_list = None
        # self.processed_files_dict_list= ColorPredictor.process_files(self.files_dict_list, self.save_intermediate_images )
        self.process_files()
        
        # Train if able to and requested 
        if self.training_queued:
            self.train()


 
    def set_train_state(self, train_if_not_trained:bool, train_regardless:bool) -> None:
        """Sets the trainable, trained and training_queued instance variables by looking if data is available,
        whether the user requested to train, and if the weights have been loaded

        Args:
            train_if_not_trained (bool): True if you only want to train if the model is untrained and could not load weights
            train_regardless (bool): True if you want to train regardless
        """

        # Is there data available to train with?
        self.trainable = True if len(filter_data(self.files_dict_list, {'dataset': 'train'})) > 0 else False 

        # Are the trained parameters available?
        self.trained = True if self.hsv_hist_means_stds is not None else False

        self.training_queued = False
        # User requested to train model only if it hasnt been trained bfore, there is data available
        if train_if_not_trained and self.trainable and not self.trained :
            self.training_queued = True

        # User requested to train model again, default train_regardless=False to save time
        if train_regardless and self.trainable:
            self.training_queued = True

        print('trainable?: ', self.trainable, ' trained?: ', self.trained, ' training queued?: ', self.training_queued)

    def save_hsv_hist_means_stds(self):
        # Save dictionary to a pickle file
        if self.hsv_hist_means_stds is None:
            print('Could not save hsv_hist_means_stds')
            return
        with open(self.hsv_hist_means_stds_path, 'wb') as file:
            pickle.dump(self.hsv_hist_means_stds, file)
    
    def load_hsv_hist_means_stds(self):
        # Save dictionary to a pickle file
        if not self.hsv_hist_means_stds_path.exists():
            # self.hsv_hist_means_stds = None
            print('hsv_hist_means_stds filepath does not exist')
            return
        with open(self.hsv_hist_means_stds_path, 'rb') as file:
            self.hsv_hist_means_stds = pickle.load(file)

    @staticmethod
    def color_masks_column_names() -> List[str]:
        """Returns a list of intermediate column names for the color masks

        Returns:
            List[str]: 
        """

        result = {}
        for color in ColorPredictor.color_names:
            result[f'{color}_mask_full'] = f'{color}_mask_full'
            result[f'{color}_mask'] = f'{color}_mask'
        return result
    
    @staticmethod
    def intermediate_columns_to_drop() -> List[str]:
        """Returns a list of all intermediate column names

        Returns:
            List[str]
        """
        result = list(ColorPredictor.color_masks_column_names().values())
        result += list(IntermediateColumnNames.to_dict().values())
        return result

    
    #@staticmethod
    def process_img(self, img: np.ndarray) -> Dict:
    #def process_img(img: np.ndarray, save_intermediate_images=False) -> Dict:
        """Extracts image features such as histograms and color occupancy.
        3 histograms are calculated for the H, S and V channels.
        A mask according to each color is geneerated using the hard-coded hsv_ranges.
        The weighted average of the mask determines the occupancy of that color.

        These features can be used however desired in a later stage.
        In this case, the average histogram for each color is used to
        determine the similarity of the image histograms with each color.
        This average is obtained during training from the train set. 

        The pixels with the lowest saturation and value are ignored, since most cars
        have some black and white pixels due to reflections and wheels etc.
        Edges are also ignored, since these tend to be wheels as well as other things.
        Since most cars take up almost the whole image, we use focal scaling, where
        pixels further from the center have a lower weight.
        The image is also cropped for the same reason.

        These features can also be used as input features to ML methods such as RFC, logistic regression,
        SVM, k-NN,  Naive Bayes and LDA. The features first need to be transformed with PCA. .

        Args:
            img (np.ndarray): RGB with range 0.0 to 1.0

        Returns:
            Dict: descriptive properties of image
        """

        # Dictionary to return image properties
        return_dict = {}

        # Intermediate Images dictionary
        intermedate_dict = {}
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.original] = deepcopy(img)

        # Cropping
        img = crop_image(img, ColorPredictor.crop_ratio)
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.cropped] = deepcopy(img)

        # Ignore the edges
        # edges_mask = get_edges_mask(img, val=160, blur_ksize=3) # If img is loaded with cv2
        edges_mask = get_edges_mask(mlib_rgb_to_cv2_rgb(img), # Covert to cv2 image
                                    val=ColorPredictor.edges_val,
                                    blur_ksize=ColorPredictor.edges_blur_ksize) # If img is loaded with mlib
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.edges_mask] = deepcopy(edges_mask)
        
        # Convert to HSV
        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = mcolors.rgb_to_hsv(img)
        
        # Ignore the darkest and lightest part of image
        sv_mask = hsv_percentile_mask(hsv_img,
                                      v_min=ColorPredictor.val_percentile_min,
                                      s_min=ColorPredictor.sat_percentile_min)
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.sv_mask] = deepcopy(sv_mask)

        # Scalar mask according to distance from focal point (center of image by default)
        focal_mask = focal_spread_mask(hsv_img.shape[0], hsv_img.shape[1], min_val=ColorPredictor.focal_spread_min_val)
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.focal_mask] = deepcopy(focal_mask)

        # Combine masks
        mask = np.logical_and(edges_mask, sv_mask)
        # mask = focal_mask * mask
        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.mask] = deepcopy(mask)
            intermedate_dict[IntermediateColumnNames.total_mask] = mask * focal_mask

        # Total masked, scaled pixels
        # n_pixels = mask.sum()
        n_pixels = (mask * focal_mask).sum()
        # print(total_pixels_dist_scaled)


        # ---------------------------------
        # Create HSV masks and determine occupancy of each color
        colors_occupancy = {}
        color_masks_dict = {}
        for col in ColorPredictor.hsv_ranges.keys():
            lower = ColorPredictor.hsv_ranges[col][0]
            upper  = ColorPredictor.hsv_ranges[col][1]

            # col_mask  = hsv_inrange(hsv_img, lower*255, upper*255) # if using cv2 hsv image
            col_mask_full  = hsv_inrange_mask(hsv_img, lower, upper) # if using mlib hsv img
            col_mask = col_mask_full * mask * focal_mask

            if self.save_intermediate_images:
                color_masks_dict[f'{col}_mask_full'] = deepcopy(col_mask_full)
                color_masks_dict[f'{col}_mask'] = deepcopy(col_mask)


            col_occupancy = col_mask.sum() / n_pixels
            colors_occupancy[f'{col}_occupancy'] = col_occupancy

        # Update image info
        return_dict = {**return_dict, **colors_occupancy}
        intermedate_dict = {**intermedate_dict, **color_masks_dict}
        # ---------------------------------

        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.val_histogram_before_equalization] =  \
            np.histogram(hsv_img[:, :, 2].flatten(), ColorPredictor.bins, density=True) 

        # Equalize Value Histogram
        # hsv_flattened[:, 2:3] = cv2.equalizeHist(hsv_flattened[:, 2]) # if using cv2 hsv img
        #hsv_flattened[:, 2:3] = cv2.equalizeHist((hsv_flattened[:, 2] * 255).astype(np.uint8)) / 255# if using mlib hsv  img
        hsv_img[:, :, 2] = cv2.equalizeHist((hsv_img[:, :, 2] * 255).astype(np.uint8)) / 255 # if using mlib hsv  img


        # Filter according to mask, we only want to calculate the histograms of relevant pixels
        x, y = np.where(mask== 1.0)
        hsv_flattened = hsv_img[x, y, :]

        # Calculate Histograms of Hue, Sat, Value
        hist_hue, bins = np.histogram(hsv_flattened[:, 0], ColorPredictor.bins, density=True)          
        hist_saturation, bins = np.histogram(hsv_flattened[:, 1], ColorPredictor.bins, density=True)
        hist_value, bins = np.histogram(hsv_flattened[:, 2], ColorPredictor.bins, density=True)

        if self.save_intermediate_images:
            intermedate_dict[IntermediateColumnNames.val_histogram] =  hist_value
            intermedate_dict[IntermediateColumnNames.sat_histogram] =  hist_saturation
            intermedate_dict[IntermediateColumnNames.hue_histogram] =  hist_hue

        # histogram_dict = {'hist_hue': hist_hue, 'hist_saturation': hist_saturation, 'hist_value':hist_value}
        histograms_dict = {ColumnNames.histograms: np.stack([hist_hue, hist_saturation, hist_value ])} # shape = 3, bins
        return_dict = {**return_dict, **histograms_dict, **intermedate_dict}
        # print(return_dict)
        return return_dict
    
    @print_execution_time
    # @staticmethod
    # def process_files(self, files_dict_list: List[Dict]) -> List[Dict]:
    def process_files(self) -> None:
        """Computes features for each image file in files_dict_list, and adds these features to the dictionary

        Args:
            files_dict_list (List[Dict]): Each dictionary only requires a file path

        """

        img_features_dict_list = []
        for d in self.files_dict_list:
            file = d['file']
            try:
                # img = cv2.imread(file)  # BGR 0..255 uint8
                # print(str(file))
                # img = plt.imread(str(file)) / 255.0  # RGB
                img = plt.imread(file) / 255.0  # RGB
                if img is None: continue
                if len(img.shape) != 3: continue
                if img.shape[2] !=3: continue # not RGB
            except:
                continue
            # full_dict = {**d, **ColorPredictor.process_img(img)}
            full_dict = {**d, **self.process_img(img)}
            img_features_dict_list.append(full_dict)
        # can covert to df
        self.processed_files_dict_list = img_features_dict_list
        # return img_features_dict_list

    @print_execution_time
    def train(self) -> None:
        """Calculate self.self.hsv_hist_means_stds, which is a dictionary with a key for 'means' and 'stds'.
        The values for each is another dictionary, with keys for each color.
        The values for each color is the mean/std histogram of shape (3, bins), which is calculated over all images of that color. 
        Essentially we learn the color distribtuion for each type of color image.

        This is also where we we can implement some traditional ML like RFC. 
        
        """

        if not self.training_queued:
            print('Training is not queued, please set the training state')
            return
        
        # Determine image features if not already done so for some reason
        if self.processed_files_dict_list is None:
            self.process_files()
            # self.processed_files_dict_list= ColorPredictor.process_files(self.files_dict_list)
        
        # Only use the training set
        processed_train_files_dict_list = filter_data(self.processed_files_dict_list, {ColumnNames.dataset: 'train'})


        known_color_histograms_means = {} # shape = (3,bins) for each color
        known_color_histograms_std = {} # shape = (3,bins) for each color
        for color_name in  ColorPredictor.color_names:
            color_files_dict_list = filter_data(processed_train_files_dict_list, {ColumnNames.true_label: color_name})
            color_histograms = np.array([d[ColumnNames.histograms] for d in color_files_dict_list]) # shape = n_color_images, 3, bins

            known_color_histograms_means[color_name] = color_histograms.mean(axis=(0))
            known_color_histograms_std[color_name] = color_histograms.std(axis=(0))

        self.hsv_hist_means_stds = {'means': known_color_histograms_means, 'stds': known_color_histograms_std }
        self.save_hsv_hist_means_stds()
        # with a valid hsv_hist_means_stds now, the value of self.trained should change to True if it was False
        self.set_train_state(train_if_not_trained= False, train_regardless=False)

    @print_execution_time
    def infer(self, save_to_file: bool=False) -> pd.DataFrame:
        """Infers by looking at smallest distance for histograms, and largest occupancy rate for HSV ranges

        Returns:
            pd.DataFrame: DataFrame containing all features and predictions
        """
        # Process if not done so already
        if self.processed_files_dict_list is None: 
            #self.processed_files_dict_list= ColorPredictor.process_files(self.files_dict_list)
            self.process_files()

        # Convert to df, histograms are no longer 
        df = pd.DataFrame(self.processed_files_dict_list).drop(columns=[ColumnNames.histograms])

        # Get occupancy labels
        occupancy_labels = largest_occupancy_label(df)
        df[ColumnNames.pred_occupancy_color] = occupancy_labels

        # Training may not have been possible, or user did not request it
        if self.trained:
            images_histograms = np.array([d[ColumnNames.histograms] for d in self.processed_files_dict_list]) # shape = n_images, 3, bins
            hist_distance = histogram_distances(images_histograms,
                                                 self.hsv_hist_means_stds['means'],
                                                 self.hsv_hist_means_stds['stds'],
                                                 ColorPredictor.hsv_color_weights)
            
            # df_hist_dist only contains columns for the distances to each color
            df_hist_dist = pd.DataFrame(hist_distance)

            # Get histogram labels
            labels = smallest_distance_label(df_hist_dist)
            df[ColumnNames.pred_hist_color] = labels

            df = pd.concat([df, df_hist_dist], axis=1)

        if save_to_file:
            # Save file 
            path = self.output_path / 'inferred.csv'

            # We dont have to save intermediate images to the .csv file
            if self.save_intermediate_images:
                df.drop(columns=ColorPredictor.intermediate_columns_to_drop()).to_csv(path)
            else:
                df.to_csv(path)
            print('Saved ', str(path))

        return df

    def evaluate(self, df=None, save_to_file:bool =False) -> Tuple[List, List]:
        """Calculates the confusion matrix and saves to file. 
        Calculates precision, recall and F1 score for each class and saves to file.
        Seperate files are stored for the train and test set.
        Seperate files are stored for the HSV range method and histogram method

        Args:
            df (_type_, optional): To evaluate. Defaults to None.

        Returns:
            Tuple[List, List]: evalution dataframes and their names
        """
        if df is None:
            df = self.infer()

        # Can only evaluate it if the true color is available
        df = df[df[ColumnNames.true_label] != "None"]

        dataframes = []
        dataframe_names = []

        for dataset in ('train', 'test', 'None'):
            df_dataset = df[df[ColumnNames.dataset] == dataset]

            if len(df_dataset) == 0: continue

            # Evaluate HSV ranges method
            conf_mat_df = confusion_matrix_df(df_dataset,
                                                true_label_column=ColumnNames.true_label,
                                                pred_label_column=ColumnNames.pred_occupancy_color)
            eval_df = create_evaluation_df(df_dataset,
                                            true_label_column=ColumnNames.true_label,
                                            pred_label_column=ColumnNames.pred_occupancy_color)

            if save_to_file:
                path = self.output_path / f'{dataset}_occupancy_conf_matrix.csv'
                conf_mat_df.to_csv(path)
                print('Saved ', str(path))
            
                path = self.output_path / f'{dataset}_occupancy_scores.csv'
                eval_df.to_csv(path)
                print('Saved ', str(path))

            dataframes += [conf_mat_df, eval_df]
            dataframe_names += [f'{dataset}_occupancy_scores.csv', f'{dataset}_occupancy_scores.csv']

            # Can only evaluate if model is trained
            # Evaluate Histograms
            if self.trained:

                conf_mat_df = confusion_matrix_df(df_dataset,
                                                   true_label_column=ColumnNames.true_label,
                                                    pred_label_column=ColumnNames.pred_hist_color)
                eval_df = create_evaluation_df(df_dataset,
                                                true_label_column=ColumnNames.true_label,
                                                pred_label_column=ColumnNames.pred_hist_color)

                if save_to_file:
                    path = self.output_path / f'{dataset}_hist_conf_matrix.csv'
                    conf_mat_df.to_csv(path)
                    print('Saved ', str(path))
                
                    path = self.output_path / f'{dataset}_hist_scores.csv'
                    eval_df.to_csv(path)
                    print('Saved ', str(path))

                dataframes += [conf_mat_df, eval_df]
                dataframe_names += [f'{dataset}_hist_conf_matrix.csv', f'{dataset}_hist_scores.csv']

        return dataframes, dataframe_names
        
                
#print(ColorPredictor.intermediate_columns_to_drop())
                
#print(IntermediateColumnNames.to_dict())
# exit()

# col_predictor = ColorPredictor(data_path=Path('./dataset'),
#                                output_path=Path('./'),
#                                 top=0,
#                                train_regardless=False,
#                                save_intermediate_images=False)

# df = col_predictor.infer(save_to_file=True)
# col_predictor.evaluate(df, save_to_file=True)



