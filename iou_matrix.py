def get_iou_matrix_from_boxes(bounding_boxes1, bounding_boxes2):

    """
    Calculate IoU matrix between two sets of bounding boxes

    Parameters
    ----------
    bounding_boxes1 [numpy.ndarray of shape (n_objects, 4)]: Bounding boxes
    bounding_boxes2 [numpy.ndarray of shape (m_objects, 4)]: Bounding boxes

    Returns
    -------
    iou_matrix [numpy.ndarray of shape (n_objects, m_objects)]: IoU matrix between two sets of bounding boxes
    """

    bounding_boxes1_x1, bounding_boxes1_y1, bounding_boxes1_x2, bounding_boxes1_y2 = np.split(bounding_boxes1, 4, axis=1)
    bounding_boxes2_x1, bounding_boxes2_y1, bounding_boxes2_x2, bounding_boxes2_y2 = np.split(bounding_boxes2, 4, axis=1)

    xa = np.maximum(bounding_boxes1_x1, np.transpose(bounding_boxes2_x1))
    ya = np.maximum(bounding_boxes1_y1, np.transpose(bounding_boxes2_y1))
    xb = np.minimum(bounding_boxes1_x2, np.transpose(bounding_boxes2_x2))
    yb = np.minimum(bounding_boxes1_y2, np.transpose(bounding_boxes2_y2))

    inter_area = np.maximum((xb - xa + 1), 0) * np.maximum((yb - ya + 1), 0)
    box_a_area = (bounding_boxes1_x2 - bounding_boxes1_x1 + 1) * (bounding_boxes1_y2 - bounding_boxes1_y1 + 1)
    box_b_area = (bounding_boxes2_x2 - bounding_boxes2_x1 + 1) * (bounding_boxes2_y2 - bounding_boxes2_y1 + 1)
    iou_matrix = inter_area / (box_a_area + np.transpose(box_b_area) - inter_area)

    return iou_matrix


def get_iou_matrix_from_masks(masks1, masks2):

    """
    Calculate IOU matrix between two sets of masks

    Parameters
    ----------
    masks1 [numpy.ndarray of shape (n_objects, height, width)]: 2d binary masks
    masks2 [numpy.ndarray of shape (m_objects, height, width)]: 2d binary masks

    Returns
    -------
    iou_matrix [numpy.ndarray of shape (n_objects, m_objects)]: IoU matrix between two sets of masks
    """

    if len(list(masks1)) == 0 or len(list(masks2)) == 0:
        print(f'empty predictions - masks1 len {len(list(masks1))}, masks2 len {len(list(masks2))}')
        return np.array([[]])

    enc_masks1 = [mask_util.encode(np.asarray(p, order='F')) for p in (masks1 > 0.5).astype(np.uint8)]
    enc_masks2 = [mask_util.encode(np.asarray(p, order='F')) for p in (masks2 > 0.5).astype(np.uint8)]
    iou_matrix = mask_util.iou(enc_masks1, enc_masks2, [0] * len(enc_masks1))

    return iou_matrix
