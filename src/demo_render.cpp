#include "demo.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace smartspray::demo {
using nlohmann::json;
namespace {
const cv::Scalar background(25, 22, 19), ink(235, 232, 224), muted(165, 161, 150);
const cv::Scalar crop(240, 188, 77), accepted(129, 211, 76), rejected(87, 128, 255);
void text(cv::Mat& canvas, const std::string& value, int x, int y,
          const cv::Scalar& color = ink, double scale = 0.55) {
    cv::putText(canvas, value, {x,y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}
std::string fixed(double value, int precision=3) {
    std::ostringstream out; out << std::fixed << std::setprecision(precision) << value; return out.str();
}
cv::Scalar color(const json& record) {
    return record.at("status") == "accepted" ? accepted :
           record.at("status") == "rejected" ? rejected : crop;
}
}
cv::Mat render_annotated(const cv::Mat& original, const json& run) {
    if (original.empty() || original.type() != CV_8UC3) throw std::runtime_error("Invalid render image");
    constexpr int left=24, top=146, panel=1024, table=1080;
    const double scale = std::min(static_cast<double>(panel)/original.cols,
                                  static_cast<double>(panel)/original.rows);
    const int width = std::max(1, static_cast<int>(std::lround(original.cols*scale)));
    const int height = std::max(1, static_cast<int>(std::lround(original.rows*scale)));
    const int rows = static_cast<int>(run.at("trace").size());
    cv::Mat canvas(std::max({top+height+65, 270+rows*25, 720}), 1720, CV_8UC3, background);
    text(canvas, "SMARTSPRAY  /  ONE IMAGE TO VIRTUAL COMMANDS", left, 36, ink, 0.8);
    text(canvas, "SIMULATION ONLY - NOT CALIBRATED  |  Predictions are not ground truth", left, 68, rejected, 0.62);
    text(canvas, "Crop: not selected    Weed + anchor: accepted (green) / rejected (orange)", left, 97);
    text(canvas, "Eight illustrative swath zones; bbox centers are not verified stem locations", left, 122, muted);
    cv::Mat resized;
    cv::resize(original, resized, {width,height}, 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(left,top,width,height)));
    for (int ch=0; ch<8; ++ch) {
        const int x=left+static_cast<int>(static_cast<double>(ch)*width/8);
        if (ch) cv::line(canvas, {x,top}, {x,top+height-1}, cv::Scalar(200,200,200), 1, cv::LINE_AA);
        cv::rectangle(canvas, {x+2,top+3}, {x+36,top+25}, background, cv::FILLED);
        text(canvas, std::to_string(ch), x+12, top+20);
    }
    text(canvas, "PREDICTIONS / TRACE", table, 155, ink, 0.7);
    text(canvas, "ID         class  score   selection / controller result", table, 182, muted, 0.5);
    text(canvas, "dN -> tN for weeds; N = original graph candidate index", table, 206, muted, 0.5);
    // Labels only occupy free rectangles. Move labels away from crowded boxes and
    // connect them to the anchor; the full ordered ledger always retains every detection.
    std::vector<cv::Rect> occupied{{left,top,width,28}};
    int row=0;
    for (const auto& record : run.at("trace")) {
        const auto& d=record.at("prediction");
        const auto box=d.at("xyxy").get<std::array<double,4>>();
        const auto point = [&](double x,double y) {
            return cv::Point(left+std::clamp(static_cast<int>(std::lround(x*scale)),0,width-1),
                             top+std::clamp(static_cast<int>(std::lround(y*scale)),0,height-1));
        };
        const auto a=point(box[0],box[1]), b=point(box[2],box[3]);
        const auto anchor=point((box[0]+box[2])/2,(box[1]+box[3])/2);
        const auto c=color(record);
        cv::rectangle(canvas,a,b,c,2,cv::LINE_AA);
        if (record.at("selected_for_mapping").get<bool>())
            cv::drawMarker(canvas,anchor,c,cv::MARKER_CROSS,13,2,cv::LINE_AA);
        const auto id=record.at("detection_id").get<std::string>();
        int baseline=0;
        auto size=cv::getTextSize(id,cv::FONT_HERSHEY_SIMPLEX,0.46,1,&baseline);
        const int lw=size.width+8, lh=21;
        cv::Rect label;
        double best=std::numeric_limits<double>::infinity();
        // Integer pixel placement only affects the presentation, never the numeric plan.
        for (int y=top+29; y+lh<=top+height; y+=lh+2)
            for (int x=left; x+lw<=left+width; x+=lw+3) {
                cv::Rect candidate(x,y,lw,lh);
                if (std::any_of(occupied.begin(),occupied.end(),[&](const auto& r){return (r&candidate).area()>0;})) continue;
                const double distance=std::hypot(x-anchor.x,y-anchor.y);
                if (distance<best) {best=distance; label=candidate;}
            }
        if (label.area()) {
            occupied.push_back(label);
            cv::line(canvas,anchor,{label.x+label.width/2,label.y+label.height/2},c,1,cv::LINE_AA);
            cv::rectangle(canvas,label,background,cv::FILLED);
            text(canvas,id,label.x+4,label.y+15,c,0.46);
        }
        const std::string status = record.at("status") == "rejected" ?
            "rejected: "+record.at("plan").at("reason").get<std::string>() :
            record.at("status") == "accepted" ?
            "accepted: ch"+std::to_string(record.at("plan").at("nozzle_index").get<int>()) :
            "not selected";
        std::ostringstream ledger;
        ledger << std::left << std::setw(9) << id << "  " << d.at("class_name").get<std::string>()
               << "  " << fixed(d.at("score").get<double>()) << "  " << status;
        text(canvas,ledger.str(),table,240+25*row++,c,0.5);
    }
    if (rows==0) text(canvas,"No detections - successful no-action run",table,240);
    text(canvas,"Original file unchanged. Full floating-point boxes, targets and reasons: run.json",
         left,top+height+30,muted,0.52);
    return canvas;
}
cv::Mat render_timeline(const json& run) {
    constexpr int w=1640, h=880, x0=190, x1=1570, y0=230, row_h=58;
    cv::Mat canvas(h,w,CV_8UC3,background);
    const auto& action=run.at("action");
    const auto capture=run.at("configuration").at("capture_time_us").get<std::int64_t>();
    const auto now=run.at("simulation").at("now_us").get<std::int64_t>();
    const auto final=action.at("final_time_us").get<std::int64_t>();
    // Subtract after widening: safe at INT64_MAX, including empty schedules.
    const long double end=static_cast<long double>(final)-capture;
    const long double span=std::max(1000.0L,end);
    const auto at=[&](std::int64_t time) {
        const long double offset=static_cast<long double>(time)-capture;
        return x0+static_cast<int>(std::llround((x1-x0)*offset/span));
    };
    text(canvas,"EIGHT CHANNELS  /  VIRTUAL COMMAND TIMELINE",28,39,ink,0.83);
    text(canvas,"SIMULATION ONLY - COMMAND ON intervals, not measured physical spraying",28,75,rejected,0.65);
    text(canvas,"Final state: ALL 8 CHANNELS OFF   |   final virtual time: "+std::to_string(final)+" us",28,110,accepted,0.65);
    text(canvas,"Capture = "+std::to_string(capture)+" us   Now = "+std::to_string(now)+" us",28,146,muted,0.56);
    const int bottom=y0+8*row_h;
    for (int tick=0; tick<=5; ++tick) {
        const int x=x0+(x1-x0)*tick/5;
        cv::line(canvas,{x,y0-20},{x,bottom},cv::Scalar(62,58,53),1);
        text(canvas,fixed(static_cast<double>(span*tick/5000),2),x-24,bottom+26,muted,0.5);
    }
    for (int ch=0; ch<8; ++ch) {
        const int y=y0+ch*row_h;
        text(canvas,"Channel "+std::to_string(ch),28,y+25,ink,0.6);
        cv::rectangle(canvas,{x0,y+7},{x1,y+39},cv::Scalar(44,40,35),cv::FILLED);
        for (const auto& interval : action.at("merged_intervals")) {
            if (interval.at("nozzle_index")!=ch) continue;
            const int on=at(interval.at("on_time_us").get<std::int64_t>());
            const int off=at(interval.at("off_time_us").get<std::int64_t>());
            cv::rectangle(canvas,{on,y+7},{std::max(on+1,off),y+39},accepted,cv::FILLED);
            std::string ids;
            for (const auto& id : interval.at("source_target_ids")) {
                if (!ids.empty()) ids+="+";
                ids+=id.get<std::string>();
            }
            int baseline=0;
            if (cv::getTextSize(ids,cv::FONT_HERSHEY_SIMPLEX,0.42,1,&baseline).width+8<=off-on)
                text(canvas,ids,on+4,y+28,background,0.42);
            else {
                const auto count=interval.at("source_target_ids").size();
                const auto short_label=std::to_string(count)+" target"+(count==1 ? "" : "s");
                if (cv::getTextSize(short_label,cv::FONT_HERSHEY_SIMPLEX,0.36,1,&baseline).width+6<=off-on)
                    text(canvas,short_label,on+3,y+27,background,0.36);
            }
        }
    }
    cv::line(canvas,{at(capture),y0-27},{at(capture),bottom},crop,2);
    cv::line(canvas,{at(now),y0-27},{at(now),bottom},rejected,2);
    text(canvas,"CAPTURE",x0,y0-47,crop,0.48);
    text(canvas,"NOW",at(now)+5,y0-24,rejected,0.48);
    text(canvas,"Virtual time since capture (ms) - integer microsecond events in run.json / events.csv",220,bottom+63);
    text(canvas,"Merged target IDs are shown when they fit; complete attribution remains in run.json.",28,827,muted,0.52);
    if (action.at("merged_intervals").empty())
        text(canvas,"NO ACTION - zero scheduled intervals / zero executed events",420,185,ink,0.6);
    return canvas;
}
} // namespace smartspray::demo
