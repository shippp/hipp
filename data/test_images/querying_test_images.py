import os

import usgsxplore

import hipp

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

PATH_AERIAL_1978 = os.path.join(SCRIPT_PATH, "1978_09_06_aerial")


def download_aerial_dataset(api: usgsxplore.API) -> None:
    if not os.path.exists(PATH_AERIAL_1978):
        output_dir = os.path.join(PATH_AERIAL_1978, "raw_images")
        api.download("aerial_combin", ["ARBCSRD00010006", "ARBCSRD00010007"], output_dir=output_dir, product_number=1)


def create_aerial_fiducials_template() -> None:
    fiducials_dir = os.path.join(PATH_AERIAL_1978, "fiducials")
    if not os.path.exists(fiducials_dir):
        preproc = hipp.AerialPreprocessing(
            os.path.join(PATH_AERIAL_1978, "raw_images"), fiducials_directory=fiducials_dir
        )
        preproc.create_fiducial_template(
            distance_around_fiducial=50,
            fiducial_coordinate=(769, 462),
            subpixel_center_coordinate=(402, 405),
            corner=True,
        )
        preproc.create_fiducial_template(
            distance_around_fiducial=70,
            fiducial_coordinate=(5030, 293),
            subpixel_center_coordinate=(563, 564),
            subpixel_distance_around_fiducial=150,
            midside=True,
        )


def main() -> None:
    username = os.getenv("USGS_USERNAME") or input("Enter your USGS username: ")
    token = os.getenv("USGS_TOKEN") or input("Enter your USGS token: ")
    api = usgsxplore.API(username, token)
    download_aerial_dataset(api)
    api.logout()

    create_aerial_fiducials_template()


if __name__ == "__main__":
    main()
