#pragma once
// =============================================================================
// image_io.h — host JPEG helpers for the GPU renderer (Module 2 / Trevor).
//   * encode_jpeg_rgba : tone-mapped framebuffer (uint32 RGBA) -> JPEG bytes
//                        (in memory, for the MJPEG stream / screenshots)
//   * load_jpeg_rgb    : decode a JPEG file -> RGB8 (for the skybox texture)
// Uses libjpeg-turbo, already a project dependency.
// =============================================================================
#include <vector>
#include <cstdint>
#include <cstdio>
#include <jpeglib.h>

// Encode a row-major W*H framebuffer of 0xAABBGGRR pixels (R in low byte) to a
// JPEG byte buffer in memory. quality in [1,100].
inline std::vector<uint8_t> encode_jpeg_rgba(const uint32_t* px, int W, int H,
                                             int quality = 85) {
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned char* out = nullptr;
    unsigned long  out_size = 0;
    jpeg_mem_dest(&cinfo, &out, &out_size);

    cinfo.image_width      = W;
    cinfo.image_height     = H;
    cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    std::vector<unsigned char> row(W * 3);
    while (cinfo.next_scanline < cinfo.image_height) {
        const uint32_t* src = px + (size_t)cinfo.next_scanline * W;
        for (int x = 0; x < W; ++x) {
            uint32_t p = src[x];
            row[x*3+0] = (unsigned char)(p & 0xFF);          // R
            row[x*3+1] = (unsigned char)((p >> 8) & 0xFF);   // G
            row[x*3+2] = (unsigned char)((p >> 16) & 0xFF);  // B
        }
        JSAMPROW rp = row.data();
        jpeg_write_scanlines(&cinfo, &rp, 1);
    }
    jpeg_finish_compress(&cinfo);

    std::vector<uint8_t> result(out, out + out_size);
    jpeg_destroy_compress(&cinfo);
    if (out) free(out);
    return result;
}

// Decode a JPEG file into a tightly-packed RGB8 buffer. Returns false on error.
inline bool load_jpeg_rgb(const char* path, std::vector<uint8_t>& rgb,
                          int& W, int& H) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    W = cinfo.output_width;
    H = cinfo.output_height;
    rgb.resize((size_t)W * H * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rp = rgb.data() + (size_t)cinfo.output_scanline * W * 3;
        jpeg_read_scanlines(&cinfo, &rp, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return true;
}
