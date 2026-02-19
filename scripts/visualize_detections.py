import cv2
import argparse
from pathlib import Path
import numpy as np

from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_img_idx_to_track_idx

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image


def compute_index(reference_timestamps, query_timestamps):
    """
    Compute indices for query timestamps in reference timestamps
    
    Args:
        reference_timestamps: Array of reference timestamps (e.g., image timestamps)
        query_timestamps: Array of query timestamps (e.g., visualization timestamps)
        
    Returns:
        Array of indices into reference_timestamps
    """
    indices = np.searchsorted(reference_timestamps, query_timestamps, side='left')
    # Clamp to valid range
    indices = np.clip(indices, 0, len(reference_timestamps) - 1)
    
    # For each query timestamp, find the closest reference timestamp
    for i, (query_t, ref_idx) in enumerate(zip(query_timestamps, indices)):
        if ref_idx > 0:
            # Check if the previous timestamp is closer
            prev_diff = abs(query_t - reference_timestamps[ref_idx - 1])
            curr_diff = abs(query_t - reference_timestamps[ref_idx])
            if prev_diff < curr_diff:
                indices[i] = ref_idx - 1
    
    return indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualization script to show bounding boxes""")
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path)
    parser.add_argument("--dataset_directory", help="Path to DSEC folder including which split.", type=Path, default="/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det/test")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int, default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.", type=int, default=5000)
    parser.add_argument("--sequence", help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b", default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output", help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.", action="store_true")
    args = parser.parse_args()

    assert args.dataset_directory.exists()
    assert args.vis_time_step_us > 0
    assert args.event_time_window_us > 0

    if args.write_to_output:
        assert (args.detections_folder / f"detections_{args.sequence}.npy").exists()
        assert args.detections_folder.exists()
        output_path = args.detections_folder / f"visualization_{args.sequence}"
        output_path.mkdir(parents=True, exist_ok=True)

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)

    t0, t1 = load_start_and_end_time(dsec_directory)

    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps)

    show_detections = args.detections_folder is not None

    if not show_detections:
        print("Did not specify detections. Just showing events and images.")

    if show_detections:
        detections_file = args.detections_folder / f"detections_{args.sequence}.npy"
        detections = np.load(detections_file)
        print(f"Loaded {len(detections)} detections for sequence {args.sequence}")
        detection_timestamps = np.unique(detections['t'])
        step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    scale = 2

    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = extract_image_by_index(dsec_directory.images.image_files_distorted, [image_index])

        # find events within time window [t-event_time_window_us, t]
        # Ensure time window is within valid bounds
        event_start_time = max(t - args.event_time_window_us, t0)
        event_end_time = min(t, t1)
        
        # Skip if time window is invalid
        if event_start_time >= event_end_time:
            # Create empty events if no valid time window
            events = {'x': np.array([]), 'y': np.array([]), 'p': np.array([])}
        else:
            try:
                events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, event_start_time, event_end_time)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not extract events for time window [{event_start_time}, {event_end_time}]: {e}")
                events = {'x': np.array([]), 'y': np.array([]), 'p': np.array([])}
        
        image = draw_events_on_image(image, events['x'], events['y'], events['p'])

        if show_detections:
            # find most recent bounding boxes
            boxes_index = step_index_to_boxes_index[step]
            boxes_timestamp = detection_timestamps[boxes_index]
            boxes = detections[detections['t'] == boxes_timestamp]

            # draw them on one image
            if len(boxes) > 0:
                image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
                                         boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)

        if args.write_to_output:
            cv2.imwrite(str(output_path / ("%06d.png" % step)), image)
            if step % 100 == 0:
                print(f"Processed frame {step}/{len(vis_timestamps)}, time: {t/1000:.1f}ms")
        else:
            cv2.imshow("DSEC Det: Visualization", image)
            key = cv2.waitKey(3)
            if key == 27:  # ESC key to exit
                break

    if not args.write_to_output:
        cv2.destroyAllWindows()
    else:
        print(f"\nVisualization completed!")
        print(f"Generated {len(vis_timestamps)} frames in {output_path}")
        print(f"To create video:")
        print(f"cd {output_path}")
        print(f"ffmpeg -framerate 10 -i %06d.png -c:v libx264 -pix_fmt yuv420p {args.sequence}_visualization.mp4")

