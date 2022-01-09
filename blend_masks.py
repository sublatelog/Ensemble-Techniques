import networkx as nx


def blend_masks(prediction_boxes, prediction_masks, iou_threshold=0.9, label_threshold=0.5, iou_method='boxes', drop_single_components=True):

    """
    Blend prediction masks of multiple models based on IoU

    Parameters
    ----------
    prediction_boxes [list of shape (n_models)]: Bounding box predictions of multiple models
    prediction_masks [list of shape (n_models)]: Mask predictions of multiple models
    iou_threshold (int): IoU threshold for blending masks (0 <= iou_threshold <= 1)
    iou_method (str): boxes or masks
    label_threshold (int): Label threshold for converting soft predictions to labels (0 <= iou_threshold <= 1)
    drop_single_components (bool): Whether to discard predictions without connections or not

    Returns
    -------
    blended_masks [numpy.ndarray of shape (n_objects, height, width)]: Blended binary masks
    """

    iou_matrices = {}

    # Create all combinations of IoU matrices from given predictions
    for i in range(len(prediction_masks)):
        for j in range(i, len(prediction_masks)):
            if i == j:
                continue

            if iou_method == 'boxes':
                iou_matrix = get_iou_matrix_from_boxes(prediction_boxes[i], prediction_boxes[j])
            elif iou_method == 'masks':
                iou_matrix = get_iou_matrix_from_masks(prediction_masks[i], prediction_masks[j])

            iou_matrices[f'{i + 1}_{j + 1}'] = iou_matrix

    # Create a graph to store connected bounding boxes
    bounding_box_graph = nx.Graph()

    # Add all masks from all models as nodes
    for model_idx, boxes in enumerate(prediction_masks, start=1):
        nodes = [f'model{model_idx}_box{box_idx}' for box_idx in np.arange(len(boxes))]
        bounding_box_graph.add_nodes_from(nodes)

    del prediction_boxes

    # Add edges between nodes with IoU >= iou_threshold
    for model_combination, iou_matrix in iou_matrices.items():
        matching_boxes_idx = np.where(iou_matrix >= iou_threshold)
        model1_idx, model2_idx = model_combination.split('_')
        edges = [(f'model{model1_idx}_box{box1}', f'model{model2_idx}_box{box2}') for box1, box2 in zip(*matching_boxes_idx)]
        bounding_box_graph.add_edges_from(edges)

    del iou_matrices
    blended_masks = []

    for connections in nx.connected_components(bounding_box_graph):
        if len(connections) == 1:
            # Skip mask if its bounding isn't connected to any other bounding box
            if drop_single_components:
                continue
            else:
                # Append mask directly if its bounding box isn't connected to any other bounding box
                model_idx, box_idx = list(connections)[0].split('_')
                model_idx = int(model_idx.replace('model', ''))
                box_idx = int(box_idx.replace('box', ''))
                blended_masks.append(prediction_masks[model_idx - 1][box_idx])
        else:
            # Blend mask with its connections and append
            blended_mask = np.zeros((520, 704), dtype=np.float32)
            for connection in connections:
                model_idx, box_idx = connection.split('_')
                model_idx = int(model_idx.replace('model', ''))
                box_idx = int(box_idx.replace('box', ''))
                # Divide soft predictions with number of connections and accumulate on blended_mask
                blended_mask += (prediction_masks[model_idx - 1][box_idx] / len(connections))
            blended_masks.append(blended_mask)

    del prediction_masks, bounding_box_graph
    blended_masks = np.stack(blended_masks)
    # Convert soft predictions to binary labels
    blended_masks = np.uint8(blended_masks >= label_threshold)

    return blended_masks


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

