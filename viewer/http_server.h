#pragma once
// =============================================================================
// http_server.h — tiny browser viewer server (Module 2 / Trevor).
//
// Default GUI path for SSH use. Built on cpp-httplib (single header).
//
// NOTE ON TRANSPORT: cpp-httplib does not implement WebSocket. Rather than pull
// in a second networking dependency (the project deliberately keeps deps to
// single-header libs), live frames are pushed as an MJPEG multipart stream
// (multipart/x-mixed-replace) consumed by a plain <img>/canvas, and the browser
// sends parameter/camera updates back as JSON via HTTP POST /control. This is
// fully bidirectional and real-time over an SSH port-forward, and needs no
// extra libraries — the practical equivalent of the WebSocket design.
//
// The server is decoupled from CUDA: the render loop wires in three callbacks.
// =============================================================================
#include <functional>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

class ViewerServer {
public:
    ViewerServer(int port, std::string index_html);
    ~ViewerServer();

    // Returns the latest tone-mapped frame as JPEG bytes (may be empty early on).
    void set_frame_provider(std::function<std::vector<uint8_t>()> f);
    // Returns a JSON string of live stats (fps, per-kernel ms, resolution, ...).
    void set_stats_provider(std::function<std::string()> f);
    // Receives the raw JSON body of a POST /control message.
    void set_control_handler(std::function<void(const std::string&)> f);

    void start();   // non-blocking: launches the listen() loop on its own thread
    void stop();
    int  port() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
