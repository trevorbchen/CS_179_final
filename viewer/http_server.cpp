// =============================================================================
// http_server.cpp — MODULE 2 (Trevor): browser viewer server implementation.
// cpp-httplib lives only in this translation unit (it is a large header).
// =============================================================================
#include "http_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "httplib.h"

struct ViewerServer::Impl {
    int                       port;
    std::string               index_html;
    httplib::Server           svr;
    std::thread               thread;
    std::atomic<bool>         running{false};

    std::function<std::vector<uint8_t>()>   frame_provider;
    std::function<std::string()>            stats_provider;
    std::function<void(const std::string&)> control_handler;
};

ViewerServer::ViewerServer(int port, std::string index_html)
    : impl_(new Impl()) {
    impl_->port = port;
    impl_->index_html = std::move(index_html);
}

ViewerServer::~ViewerServer() { stop(); }

void ViewerServer::set_frame_provider(std::function<std::vector<uint8_t>()> f) {
    impl_->frame_provider = std::move(f);
}
void ViewerServer::set_stats_provider(std::function<std::string()> f) {
    impl_->stats_provider = std::move(f);
}
void ViewerServer::set_control_handler(std::function<void(const std::string&)> f) {
    impl_->control_handler = std::move(f);
}
int ViewerServer::port() const { return impl_->port; }

void ViewerServer::start() {
    using namespace std::chrono_literals;
    Impl* d = impl_.get();

    // Single-page UI.
    d->svr.Get("/", [d](const httplib::Request&, httplib::Response& res) {
        res.set_content(d->index_html, "text/html");
    });

    // Live MJPEG stream: one JPEG per multipart part, pushed ~30 fps.
    d->svr.Get("/stream", [d](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [d](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::vector<uint8_t> jpeg =
                    d->frame_provider ? d->frame_provider() : std::vector<uint8_t>{};
                if (jpeg.empty()) {                 // no frame yet — keep the stream open
                    std::this_thread::sleep_for(10ms);
                    return true;
                }
                char header[160];
                int n = std::snprintf(header, sizeof(header),
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                    jpeg.size());
                if (!sink.write(header, n)) return false;            // client gone
                if (!sink.write(reinterpret_cast<const char*>(jpeg.data()),
                                jpeg.size())) return false;
                if (!sink.write("\r\n", 2)) return false;
                std::this_thread::sleep_for(33ms);                   // ~30 fps cap
                return true;
            });
    });

    // Single still frame (handy for headless tests / curl).
    d->svr.Get("/snapshot", [d](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> jpeg =
            d->frame_provider ? d->frame_provider() : std::vector<uint8_t>{};
        res.set_content(reinterpret_cast<const char*>(jpeg.data()), jpeg.size(),
                        "image/jpeg");
    });

    // Live stats (FPS + per-kernel timings) as JSON.
    d->svr.Get("/stats", [d](const httplib::Request&, httplib::Response& res) {
        std::string s = d->stats_provider ? d->stats_provider() : std::string("{}");
        res.set_content(s, "application/json");
    });

    // Parameter / camera updates from the browser.
    d->svr.Post("/control", [d](const httplib::Request& req, httplib::Response& res) {
        if (d->control_handler) d->control_handler(req.body);
        res.set_content("{\"ok\":true}", "application/json");
    });

    d->running = true;
    d->thread = std::thread([d]() {
        // 0.0.0.0 so a VSCode-forwarded localhost connection on the remote works.
        if (!d->svr.listen("0.0.0.0", d->port)) {
            std::fprintf(stderr, "[viewer] failed to bind port %d\n", d->port);
        }
    });
    // Give listen() a moment to come up so the startup banner is accurate.
    std::this_thread::sleep_for(150ms);
    std::printf("[viewer] serving on http://localhost:%d/\n", d->port);
}

void ViewerServer::stop() {
    if (!impl_ || !impl_->running) return;
    impl_->running = false;
    impl_->svr.stop();
    if (impl_->thread.joinable()) impl_->thread.join();
}
