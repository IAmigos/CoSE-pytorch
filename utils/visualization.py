import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.switch_backend('agg')
import numpy as np
import pickle
# import tensorflow as tf


def get_min_max(values, offset_ratio=0.0):
    min_ = values.min()
    max_ = values.max()
    offset_ = max(abs(min_), abs(max_))
    min_ -= offset_ * offset_ratio
    max_ += offset_ * offset_ratio
    return (min_, max_)


def transform_strokes_to_image(drawing, output_path, output_file, seq_len_drawing, start_coord_drawing, mean_channel,
                               std_channel, num_strokes=None, square_figure=False, x_borders=None, y_borders=None,
                               colors=None, marker_size=0, alpha=1.0, highlight_start=False):
    """
    Args:
        drawing: a diagram sample with shape (max_num_stroke)
        output_path: output path where file be saved
        output_file: name of output file without extension
        seq_len_drawing: amount of points for each stroke
        start_coord_drawing: start coord for each stroke
        mean_channel: means for x and y
        std_channel: std for x and y
        num_strokes: # of strokes to be rendered (stroke_list[:num_strokes]).
        square_figure: if image will be reshaped to a square figure
        x_borders: a tuple of min, max x coordinates.
        y_borders: a tuple of min, max y coordinates.
        colors: list of colors per stroke.
        marker_size: marker size for ploting points in the strokes
        alpha: opacity of drawing
        highlight_start: if order of strokes and highlight of first point of stroke will be ploted 
    Returns: fig, ax
    """
    
    save_path = os.path.join(output_path, output_file)
    
    if drawing.shape[2] == 2:
        drawing = np.concatenate([drawing, np.zeros((drawing.size(0), drawing.size(1), 1))], axis = -1)
    
    # unnormalize
    mean = np.concatenate([std_channel, np.array([0])])
    std = np.concatenate([std_channel, np.array([0])])
    drawing = drawing * std + mean

    stroke_list = []
    for i in range(drawing.shape[0]):
        start_coords = np.concatenate(
            [start_coord_drawing[i].squeeze(), np.array([0])])
        i_stroke = drawing[i][:seq_len_drawing[i]] + start_coords
        stroke_list.append(i_stroke)

    if num_strokes:
        stroke_list = stroke_list[:num_strokes]

    # render
    if len(stroke_list) > 1:
        all_strokes = np.concatenate(stroke_list, axis=0)
    else:
        all_strokes = stroke_list[0]
    if all_strokes.shape[0] == 0:
        return None

    if x_borders is None:
        x_borders = get_min_max(all_strokes[:, 0], 0.1)
    if y_borders is None:
        y_borders = get_min_max(all_strokes[:, 1], 0.1)

    # Set figure size dynamically. Max resolution is 2000 pixels.
    y_range = abs(y_borders[0]) + abs(y_borders[1])
    x_range = abs(x_borders[0]) + abs(x_borders[1])
    base_size = 2
    max_size = 20

    if y_range / x_range > 4 or x_range / y_range > 4:
        x_size = base_size if x_range < y_range else min(
            max_size, base_size * x_range / y_range)
        y_size = base_size if y_range < x_range else min(
            max_size, base_size * y_range / x_range)
    else:
        x_size = max((x_range / (y_range + x_range)) * max_size, base_size)
        y_size = max((y_range / (y_range + x_range)) * max_size, base_size)

    if square_figure:
        max_xysize = max(x_size, y_size)
        x_size, y_size = max_xysize, max_xysize
    fig, ax = plt.subplots(figsize=(x_size, y_size))

    plt.axis("tight")
    plt.axis('off')
    ax.set_xlim(x_borders)
    ax.set_ylim(y_borders)

    for i, stroke in enumerate(stroke_list):
        color = colors[i] if colors is not None else mpl.cm.tab20.colors[i % 20]
        if marker_size > 0:
            ax.plot(stroke[:, 0], stroke[:, 1], lw=3,
                    color=color, marker='o', markersize=marker_size)
        else:
            plt_stroke = ax.plot(
                stroke[:, 0], stroke[:, 1], lw=3, color=color, alpha=alpha)

            if highlight_start:
                plt.plot(stroke[0, 0], stroke[0, 1], 'ro',
                         lw=3, markersize=12, color=color)
                mean_pos = stroke.mean(0)
                text_x = mean_pos[0]
                text_y = mean_pos[1]
                on_stroke = np.any(np.linalg.norm(
                    stroke[:, 0:2] - mean_pos[np.newaxis, :2], axis=1) < 0.05)
                if on_stroke:
                    mean_pos = stroke[:stroke.shape[0] // 3].mean(0)
                    text_x = mean_pos[0]
                    text_y = mean_pos[1]
                    text_x -= (text_y - stroke[0, 1]) / 2.0
                    text_y -= (text_x - stroke[0, 0]) / 2.0
                ax.text(text_x, text_y, str(i + 1), fontsize=25,
                        ha='center', va='center', color=plt_stroke[0].get_color())

    if fig is not None:
        fig.savefig(save_path + ".png", format="png")

        # with tf.io.gfile.GFile(save_path + ".png", "w") as tf_save_path:
        #     fig.savefig(tf_save_path, format="png", bbox_inches='tight', dpi=200)
        #     plt.close()

        # with open(save_path + ".svg", "w") as tf_save_path:
        #     fig.savefig(tf_save_path, format='svg', dpi=300)
        #     plt.close()

    return fig, ax


if __name__ == '__main__':
    inputs_list = 'inputs_list_based_x16.pkl'
    target_list = 'target_list_based_x16.pkl'
    inputs_dic = 'inputs_dict_based.pkl'
    target_dic = 'target_dict_based.pkl'
    stats_json = 'didi_wo_text-stats-origin_abs_pos.json'
    path = '/data/ajimenez/cose/train/'
    stats_path = '/data/jcabrera/didi_wo_text/'
    filenames = {
        "inputs_file": inputs_list,
        "targets_file": target_list,
        "stats_file": stats_json
    }

    inputs = pickle.load(
        open(os.path.join(path, filenames["inputs_file"]), 'rb'))
    targets = pickle.load(
        open(os.path.join(path, filenames["targets_file"]), 'rb'))
    with open(os.path.join(stats_path, filenames["stats_file"])) as json_file:
        stats = json.load(json_file)
        
    log_dir = '/home/dibanez/pruebas'  # TODO definir log_dir

    input_sample = inputs[1]

    i_diagram = 7
    max_num_strokes = input_sample['num_strokes'].max()
    drawing_sample = input_sample['encoder_inputs'][max_num_strokes *
                                                    i_diagram: max_num_strokes*i_diagram + max_num_strokes]
    seq_len_drawing = input_sample['seq_len'][max_num_strokes *
                                              i_diagram: max_num_strokes*i_diagram + max_num_strokes]
    num_strokes_drawing = input_sample['num_strokes'][i_diagram]
    start_coord_drawing = input_sample['start_coord'][max_num_strokes *
                                                      i_diagram: max_num_strokes*i_diagram + max_num_strokes]
    mean_channel = stats['mean_channel'][:2]
    std_channel = np.sqrt(stats['var_channel'][:2])
    
    # fig, _ = transform_strokes_to_image(drawing_sample, 'noutput_pruebas', seq_len_drawing, start_coord_drawing,
    #                                     mean_channel, std_channel, num_strokes_drawing, square_figure=True)
    fig, _ = transform_strokes_to_image(drawing_sample, log_dir, 'noutput_pruebas', seq_len_drawing, start_coord_drawing,
                                        mean_channel, std_channel, num_strokes_drawing, square_figure=True, alpha=0.5, highlight_start=True)
    fig.show()
