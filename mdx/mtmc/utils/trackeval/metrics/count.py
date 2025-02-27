from ._base_metric import _BaseMetric
from mdx.mtmc.utils.trackeval import _timing


class Count(_BaseMetric):
    """
    Class which simply counts the number of tracker and gt detections and ids.
    
    :param Dict config: configuration for the app
    ::

        identity = trackeval.metrics.Count(config)
    """
    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']
        self.fields = self.integer_fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """
        Returns counts for one sequence
        
        :param Dict data: dictionary containing the data for the sequence
        
        :return: dictionary containing the calculated count metrics
        :rtype: Dict[str, Dict[str]]
        """
        # Get results
        res = {'Dets': data['num_tracker_dets'],
               'GT_Dets': data['num_gt_dets'],
               'IDs': data['num_tracker_ids'],
               'GT_IDs': data['num_gt_ids'],
               'Frames': data['num_timesteps']}
        return res

    def combine_sequences(self, all_res):
        """
        Combines metrics across all sequences
        
        :param Dict[str, float] all_res: dictionary containing the metrics for each sequence        
        :return: dictionary containing the combined metrics across sequences
        :rtype: Dict[str, float]
        """
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        """
        Combines metrics across all classes by averaging over the class values
        
        :param Dict[str, float] all_res: dictionary containing the ID metrics for each class
        :param bool ignore_empty_classes: Flag to ignore empty classes, defaults to False
        :return: dictionary containing the combined metrics averaged over classes
        :rtype: Dict[str, float]
        """
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        return res

    def combine_classes_det_averaged(self, all_res):
        """
        Combines metrics across all classes by averaging over the detection values
        
        :param Dict[str, float] all_res: dictionary containing the metrics for each class        
        :return: dictionary containing the combined metrics averaged over detections
        :rtype: Dict[str, float]
        """
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        return res