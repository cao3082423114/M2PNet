import cv2
import numpy as np
from PIL import Image
import os
import glob as gb
from tqdm import tqdm

flow_path = "./data/SegTrackv2/FlowImages_gap-1"

flows_basename = [os.path.basename(x) for x in gb.glob(os.path.join(flow_path, '*').replace("\\", "/"))]

for _ in flows_basename:
    flow_type = os.path.join(flow_path, _).replace("\\", "/")
    # Creating an output folder
    # save_flow = os.path.join("../data/DAVIS2016/flow_resimg/output_flow_gap-1", _).replace("\\", "/")
    save_res = os.path.join("./data/SegTrackv2/flow_resimg/residualsImages_gap-1", _).replace("\\", "/")
    # if not os.path.exists(save_flow):
    #     os.makedirs(save_flow)
    # else:
    #     continue
    if not os.path.exists(save_res):
        os.makedirs(save_res)
    # else:
    #     continue

    # Creating an output folder
    # save_flow_2 = os.path.join("../data/DAVIS2016/flow_resimg/output_flow_gap-2", _).replace("\\", "/")
    save_res_2 = os.path.join("./data/SegTrackv2/flow_resimg/residualsImages_gap-2", _).replace("\\", "/")
    # if not os.path.exists(save_flow_2):
    #     os.makedirs(save_flow_2)
    if not os.path.exists(save_res_2):
        os.makedirs(save_res_2)


    for flo in tqdm(sorted(os.listdir(flow_type)), desc="{}".format(_), leave=False):
        num_file = len(os.listdir(flow_type))
        flo_next = int(flo.split('.')[0])
        flo_next=flo_next+1
        if flo_next == num_file+1:
            break
        flo_next = str(flo_next).zfill(5)
        flo_path = os.path.join(flow_type, "{}.png").format(flo_next).replace("\\", "/")
        flo_next_path = os.path.join(flow_type, "{}.png").format(flo_next).replace("\\", "/").replace("FlowImages_gap-1",
                                                                                                      "FlowImages_gap-2")
        flow_image = Image.open(flo_path)
        flow_image_next = Image.open(flo_next_path)

        gray_image = flow_image.convert("L")
        gray_array = np.array(gray_image)
        gray_image_2 = flow_image_next.convert("L")
        gray_array_2 = np.array(gray_image_2)

        flo_residual = cv2.absdiff(gray_array, gray_array_2)

        # cv2.imshow('flo_residual', flo_residual)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # cv2.imshow('res_residual', res_residual)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        diff_image = Image.new("L", gray_image.size)
        flo_residual = flo_residual.transpose()
        for i in range(gray_image.size[0]):
            for j in range(gray_image.size[1]):
                pixel_value = int(gray_image.getpixel((i, j))  - flo_residual[i, j] * 10)
                diff_image.putpixel((i, j), pixel_value)
        diff_image.save("{}/{}".format(save_res, ("{}.png").format(flo_next)))

        # final_diff_flow = Image.new("RGB", gray_image.size)
        # for i in range(flow_image.size[0]):
        #     for j in range(flow_image.size[1]):
        #         r, g, b = flow_image.getpixel((i, j))
        #         gray_value = diff_image.getpixel((i, j))
        #         final_pixel_value = tuple(
        #             map(int, (r * gray_value / 255, g * gray_value / 255, b * gray_value / 255)))
        #         final_diff_flow.putpixel((i, j), final_pixel_value)
        # final_diff_flow.save("{}/{}".format(save_flow, ("{}.png").format(flo_next)))

        diff_image_2 = Image.new("L", gray_image_2.size)
        for i in range(gray_image_2.size[0]):
            for j in range(gray_image_2.size[1]):
                pixel_value_2 = int(gray_image_2.getpixel((i, j)) - flo_residual[i, j] * 10)
                diff_image_2.putpixel((i, j), pixel_value_2)
        diff_image_2.save("{}/{}".format(save_res_2, ("{}.png").format(flo_next)))

        # final_diff_flow_2 = Image.new("RGB", gray_image_2.size)
        # for i in range(flow_image_next.size[0]):
        #     for j in range(flow_image_next.size[1]):
        #         r_2, g_2, b_2 = flow_image_next.getpixel((i, j))
        #         gray_value = diff_image.getpixel((i, j))
        #         final_pixel_value = tuple(
        #             map(int, (r_2 * gray_value / 255, g_2 * gray_value / 255, b_2 * gray_value / 255)))
        #         final_diff_flow_2.putpixel((i, j), final_pixel_value)
        # final_diff_flow_2.save("{}/{}".format(save_flow_2, ("{}.png").format(flo_next)))
