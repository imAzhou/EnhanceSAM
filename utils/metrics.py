import cv2
import numpy as np
import torch
from mmdet.evaluation.metrics import SemSegMetric,CocoPanopticMetric,CocoMetric
from prettytable import PrettyTable
import torch
from skimage.measure import label
from scipy.ndimage import binary_dilation
from mmengine.logging import MMLogger

class InstMetric:
    def __init__(self) -> None:
        super().__init__()
        self.aji_scores = []
        self.iou_scores = []
        self.dice_scores = []

    def find_connect(self, binary_mask: torch.Tensor, dilation = False):

        image_mask_np = binary_mask.cpu().numpy().astype(np.uint8)
        # labels: 0 mean background, 1...n mean different connection region
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=4)
        if dilation and np.sum(labels) > 0:
            iid = np.unique(labels)
            for i in iid[1:]:
                mask = labels == i
                dilation_mask = binary_dilation(mask)
                labels[dilation_mask] = i
        return labels

    def AJI_fast(self, gt_mask, pred_mask):
        '''
        Requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
        not [2, 3, 6, 10].
        Args:
            - gt_mask: (np.array), shape is (h,w), pixel value > 0 means instance IDs, 0 mean background
            - pred_mask: (np.array), shape is (h,w), pixel value means same as gt_mask
        '''
        gs, g_areas = np.unique(gt_mask, return_counts=True)
        assert np.all(gs == np.arange(len(gs)))
        ss, s_areas = np.unique(pred_mask, return_counts=True)
        assert np.all(ss == np.arange(len(ss)))

        i_idx, i_cnt = np.unique(np.concatenate([gt_mask.reshape(1, -1), pred_mask.reshape(1, -1)]),
                                return_counts=True, axis=1)
        i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int32)
        i_arr[i_idx[0], i_idx[1]] += i_cnt
        u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
        iou_arr = 1.0 * i_arr / u_arr

        i_arr = i_arr[1:, 1:]
        u_arr = u_arr[1:, 1:]
        iou_arr = iou_arr[1:, 1:]

        # no pred inst
        if iou_arr.shape[1] == 0:
            return 0.
        j = np.argmax(iou_arr, axis=1)

        c = np.sum(i_arr[np.arange(len(gs) - 1), j])
        u = np.sum(u_arr[np.arange(len(gs) - 1), j])
        used = np.zeros(shape=(len(ss) - 1), dtype=np.int32)
        used[j] = 1
        u += (np.sum(s_areas[1:] * (1 - used)))
        return 1.0 * c / u

    def inst_iou_dice(self, gt, pred):
        """ Compute the object-level metrics between predicted and
        groundtruth: dice, iou """

        # get connected components
        pred_labeled = label(pred, connectivity=2)
        Ns = len(np.unique(pred_labeled)) - 1
        gt_labeled = label(gt, connectivity=2)
        Ng = len(np.unique(gt_labeled)) - 1

        # --- compute dice, iou, hausdorff --- #
        pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
        gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

        # compute how well groundtruth object overlaps its segmented object
        dice_g = 0.0
        iou_g = 0.0
        for i in range(1, Ng + 1):
            gt_i = np.where(gt_labeled == i, 1, 0)
            overlap_parts = gt_i * pred_labeled
            # get intersection objects numbers in image
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]
            gamma_i = float(np.sum(gt_i)) / gt_objs_area
            if obj_no.size == 0:   # no intersection object
                dice_i = 0
                iou_i = 0
            else:
                # find max overlap object
                obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
                seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
                pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object
                overlap_area = np.max(obj_areas)  # overlap area
                dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
                iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
            dice_g += gamma_i * dice_i
            iou_g += gamma_i * iou_i
            
        # compute how well segmented object overlaps its groundtruth object
        dice_s = 0.0
        iou_s = 0.0
        for j in range(1, Ns + 1):
            pred_j = np.where(pred_labeled == j, 1, 0)
            overlap_parts = pred_j * gt_labeled
            # get intersection objects number in gt
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]
            sigma_j = float(np.sum(pred_j)) / pred_objs_area
            # no intersection object
            if obj_no.size == 0:
                dice_j = 0
                iou_j = 0
            else:
                # find max overlap gt
                gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
                gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
                gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

                overlap_area = np.max(gt_areas)  # overlap area
                dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
                iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)
            dice_s += sigma_j * dice_j
            iou_s += sigma_j * iou_j

        return (iou_g + iou_s) / 2, (dice_g + dice_s) / 2

    def process(self, gt_mask, binary_mask: torch.Tensor,dilation = False):
        pred_mask = self.find_connect(binary_mask,dilation)
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = np.array(gt_mask)

        aji_score = self.AJI_fast(gt_mask, pred_mask)
        iou_score,dice_score = self.inst_iou_dice(gt_mask, pred_mask)
        self.aji_scores.append(aji_score)
        self.iou_scores.append(iou_score)
        self.dice_scores.append(dice_score)
    
    def evaluate(self, total_cnts):
        logger: MMLogger = MMLogger.get_current_instance()
        table_data = PrettyTable()
        metrics = dict()

        average_aji = np.sum(np.array(self.aji_scores)) / total_cnts
        self.aji_scores = []
        metrics['AJI'] = average_aji

        average_iou = np.sum(np.array(self.iou_scores)) / total_cnts
        self.iou_scores = []
        metrics['IoU'] = average_iou

        average_dice = np.sum(np.array(self.dice_scores)) / total_cnts
        self.dice_scores = []
        metrics['Dice'] = average_dice
        
        for key,value in metrics.items():
            table_data.add_column(key,[f'{value:.4f}'])
        logger.info('\n' + table_data.get_string())

        return metrics


def get_metrics(types:list[str], metainfo, restinfo: dict = None):
    '''can choice metric: ['pixel', 'inst']
        pixel:  semantic_evaluator(mIoU, mFscore)
        inst: panoptic_evaluator(bPQ, mPQ...), AJI_evaluator
    '''
    evaluators = {}

    # if 'iou' in types:
    #     evaluators['iou_evaluator'] = IoUMetric(iou_metrics=['mIoU','mFscore'],logger=logger)
    #     evaluators['iou_evaluator'].dataset_meta = metainfo

    if 'pixel' in types:
        evaluators['semantic_evaluator'] = SemSegMetric(iou_metrics=['mIoU', 'mFscore'])
        evaluators['semantic_evaluator'].dataset_meta = metainfo

    if 'inst' in types:
        evaluators['panoptic_evaluator'] = CocoPanopticMetric(
            ann_file = restinfo['panoptic_ann_file'],
            seg_prefix = restinfo['seg_prefix'],
            classwise = True
        )
        evaluators['panoptic_evaluator'].dataset_meta = metainfo
        # evaluators['detection_evaluator'] = CocoMetric(
        #     ann_file = restinfo['detection_ann_file'],
        #     metric = ['bbox', 'segm'],
        #     classwise = True
        # )
        # evaluators['detection_evaluator'].dataset_meta = metainfo

        evaluators['inst_evaluator'] = InstMetric()

    return evaluators