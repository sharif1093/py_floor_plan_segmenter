from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pathlib import Path

import base64
import yaml
import zlib
import cv2
import io

from py_floor_plan_segmenter.segment import load_gray_map_from_buffer
from py_floor_plan_segmenter.modules import do_segment


app = FastAPI()


def map_encode(file):
    with open(file, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf8')


@app.post("/processfile/", response_class=StreamingResponse)
async def create_process_file(uploaded_file: UploadFile):
    """Reads an input map file in .pgm.nz format.

    Todo:
        Later, we shall only send the uuid to the rank map.
        Also, we may upload the result back to the Orbital server.

    Args:
        file (UploadFile): Should be a .pgm.nz file.

    Returns:
        File: Returns a .pgm.nz file.
    """
    #######################
    # Read the input file #
    #######################
    contents = await uploaded_file.read()
    inflated = zlib.decompress(contents)
    # NOTE: To write the file on the disk:
    # with open(file_location, "wb+") as file_object:
    #     file_object.write(contents)
    # with gzip.open(new_file, "wb", compresslevel=1) as fh:
    #     inflated = zlib.decompress(contents)
    #     fh.write(inflated)

    # NOTE: Write the files to the disk for debugging
    # cv2.imwrite('file/rank.png', np.uint8(rank*255))
    # cv2.imwrite('file/rank.pgm', np.uint8(rank*255))

    ##############
    # Processing #
    ##############
    config_file = Path(__file__).parent.absolute() / "default.yml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    raw = load_gray_map_from_buffer(inflated)
    # cv2.imwrite('file/rank.pgm', np.uint8(raw*255))
    segments = do_segment(raw, **config)
    # cv2.imwrite('file/segments.png', segments)

    ##########################
    # Writing output to file #
    ##########################
    # See: https://stackoverflow.com/questions/52865771/write-opencv-image-in-memory-to-bytesio-or-tempfile
    is_success, buffer = cv2.imencode(".pgm", segments)
    if not is_success:
        raise HTTPException(status_code=404, detail="Image encoding failed")
    deflated = zlib.compress(buffer, level=9)

    # media_type: https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    #             https://www.iana.org/assignments/media-types/media-types.xhtml
    attachment = StreamingResponse(io.BytesIO(
        deflated), media_type="application/zlib")
    outfile_name = Path(Path(uploaded_file.filename).stem).stem
    attachment.headers["Content-Disposition"] = f"attachment; filename={outfile_name}_segments.pgm.nz"

    return attachment
