#include "demo.hpp"
#include "sha256.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using namespace smartspray;
using nlohmann::json;
namespace {
void check(bool value,const std::string& why) { if (!value) throw std::runtime_error(why); }
template<class F> void fails(F f) {
    bool failed=false; try {f();} catch(const std::exception&) {failed=true;}
    check(failed,"Expected explicit failure");
}
vision::Detection detection(std::size_t id,int cls,float x=640,float y=512) {
    return {cls,0.8F,{x-1,y-1,x+1,y+1},id};
}
json run(std::vector<vision::Detection> d={},simulation::Profile p={}) {
    return demo::compose(d,vision::letterbox_geometry(2048,1024),p);
}
void off(const json& r) {
    check(r["action"]["schedule"]==r["action"]["executed_events"],"Actual executor log differs");
    check(r["action"]["final_states"]==json::array({"OFF","OFF","OFF","OFF","OFF","OFF","OFF","OFF"}),"Not all OFF");
    check(r["action"]["final_time_us"].get<std::int64_t>()>=r["simulation"]["now_us"].get<std::int64_t>(),"Final time before now");
}
void selection_mapping() {
    auto r=run({detection(91,0),detection(17,1)});
    check(r["counts"]["targets"]==1 && r["counts"]["crops_not_selected"]==1,"Selection");
    check(r["trace"][0]["status"]=="not_selected" && !r["trace"][0].contains("plan"),"Crop became controller failure");
    check(r["trace"][1]["target_id"]=="t17" && r["trace"][1]["detection_id"]=="d17","IDs");
    check(r["trace"][1]["anchor"]["u_px"]==640.0 && r["trace"][1]["anchor"]["v_px"]==512.0,"Original center");
    check(r["trace"][1]["target"]["x_m"]==0.625 && r["trace"][1]["target"]["forward_m"]==1.0,"Mapping");
    check(r["trace"][1]["plan"]["nozzle_index"]==2 && r["trace"][1]["plan"]["on_time_us"]==1450000,"Plan reference");
    off(r);
}
void all_boundaries() {
    for (int i=1;i<8;++i) for (int delta=-1;delta<=1;++delta) {
        auto r=run({detection(3,1,static_cast<float>(i*256+delta))});
        check(r["trace"][0]["plan"]["nozzle_index"]==(delta<0?i-1:i),"Channel boundary");
    }
    auto l=run({{1,.8F,{0,0,2,2},1}});
    auto r=run({{1,.8F,{2046,0,2048,2},1}});
    check(l["trace"][0]["plan"]["nozzle_index"]==0 && r["trace"][0]["plan"]["nozzle_index"]==7,"Image edge channels");
    auto bottom=run({{1,.8F,{0,1022,2,1024},1}});
    check(bottom["trace"][0]["plan"]["reason"]=="TOO_LATE","Bottom edge");
}
void image_limits() {
    for (const auto box:std::vector<std::array<float,4>>{{-1,0,1,1},{0,-1,1,1},{2047,0,2049,1},
        {0,1023,1,1025},{2,0,1,1},{0,2,1,1},{0,0,0,1},{0,0,1,0},
        {0,0,std::numeric_limits<float>::infinity(),1},{0,0,1,std::numeric_limits<float>::quiet_NaN()}}) {
        fails([&]{run({{1,.8F,box,1}});});
    }
    fails([]{demo::compose({},vision::letterbox_geometry(0,1024),{});});
    fails([]{simulation::map_and_execute({{"original_width",1e300},{"original_height",1024},{"detections",json::array()}});});
}
void stable_ids_merging() {
    simulation::Profile p; p.controller.pulse_duration_us=125000;
    auto a=run({detection(90,1),detection(20,1,640,384),detection(50,1,640,448)},p);
    auto b=run({detection(50,1,640,448),detection(90,1),detection(20,1,640,384)},p);
    check(a["action"]["merged_intervals"]==b["action"]["merged_intervals"],"Stable merge");
    check(a["action"]["merged_intervals"][0]["source_target_ids"]==json::array({"t20","t50","t90"}),"All source IDs");
    check(a["action"]["target_results"].size()==3 && a["action"]["merged_intervals"].size()==1,"Original pulses retained");
    check(a["action"]["merged_intervals"][0]["on_time_us"]==1450000 &&
          a["action"]["merged_intervals"][0]["off_time_us"]==1700000,"Touch/overlap union");
    off(a);
}
void empty() {auto r=run(); check(r["counts"]["events"]==0,"Empty events"); off(r);}
void crops_only() {auto r=run({detection(1,0),detection(2,0)}); check(r["counts"]["events"]==0,"Crop events"); off(r);}
void all_rejected() {
    auto r=run({detection(1,1,640,1000),detection(2,1,320,1000)});
    check(r["counts"]["accepted"]==0 && r["counts"]["rejected"]==2 && r["counts"]["events"]==0,"All rejected"); off(r);
}
void mixed() {
    auto r=run({detection(1,1),detection(2,1,640,1000),detection(3,0)});
    check(r["counts"]["accepted"]==1 && r["counts"]["rejected"]==1 && r["counts"]["events"]==2,"Mixed counts"); off(r);
}
void deadline() {
    simulation::Profile p; p.controller.actuator_delay_us=0; p.simulated_processing_delay_us=500000;
    auto exact=run({detection(1,1)},p);
    check(exact["trace"][0]["plan"]["on_time_us"]==exact["simulation"]["now_us"],"Equality accepted");
    ++p.simulated_processing_delay_us;
    auto late=run({detection(1,1)},p);
    check(late["trace"][0]["plan"]["reason"]=="TOO_LATE" && late["counts"]["events"]==0,"One us miss");
}
void time_limits() {
    simulation::Profile p;
    p.capture_time_us=std::numeric_limits<std::int64_t>::max()-1000000;
    p.simulated_processing_delay_us=500000; p.controller.actuator_delay_us=0;
    p.controller.pulse_duration_us=500000;
    auto r=run({detection(1,1)},p);
    check(r["action"]["final_time_us"]==std::numeric_limits<std::int64_t>::max(),"No +1 at INT64_MAX");
    off(r); check(!demo::render_timeline(r).empty(),"Render limit");
    p.capture_time_us=std::numeric_limits<std::int64_t>::max(); p.simulated_processing_delay_us=0;
    auto e=run({},p); off(e); check(!demo::render_timeline(e).empty(),"Empty at max");
    auto rejected=run({detection(1,1)},p);
    check(rejected["trace"][0]["plan"]["reason"]=="TIME_OUT_OF_RANGE","Controller overflow rejection"); off(rejected);
    p.simulated_processing_delay_us=1; fails([&]{run({},p);});
}
void config_validation() {
    const auto valid=simulation::profile_json({});
    check(simulation::validate_profile(simulation::parse_profile(valid))==1100000,"M3 defaults");
    for (const char* key:{"nozzle_pitch_m","speed_mps","pulse_duration_us","lookahead_m"}) {
        auto j=valid; j[key]=0; fails([&]{simulation::parse_profile(j);});
    }
    for (const char* key:{"capture_time_us","simulated_processing_delay_us","actuator_delay_us"}) {
        auto j=valid; j[key]=-1; fails([&]{simulation::parse_profile(j);});
    }
    for (const char* key:{"capture_time_us","actuator_delay_us","schema_version"}) {
        auto j=valid; j[key]=1.5; fails([&]{simulation::parse_profile(j);});
        j[key]=std::numeric_limits<std::uint64_t>::max(); fails([&]{simulation::parse_profile(j);});
    }
    auto j=valid; j["extra"]=1; fails([&]{simulation::parse_profile(j);});
    j=valid; j.erase("lookahead_m"); fails([&]{simulation::parse_profile(j);});
    j=valid; j["schema_version"]=2; fails([&]{simulation::parse_profile(j);});
    j=valid; j["speed_mps"]=true; fails([&]{simulation::parse_profile(j);});
    simulation::Profile p; p.lookahead_m=std::numeric_limits<double>::infinity(); fails([&]{run({},p);});
    p={}; p.controller.nozzle_pitch_m=std::numeric_limits<double>::max(); fails([&]{run({},p);});
}
void invalid_detection() {
    fails([]{run({detection(1,1),detection(1,0)});});
    auto d=detection(1,1); d.score=.25; fails([&]{run({d});});
    d=detection(1,2); fails([&]{run({d});});
}
void json_csv() {
    auto r=run({detection(1,1),detection(2,1,320)});
    const auto parsed=json::parse(r.dump());
    check(parsed==r,"JSON roundtrip");
    check(demo::events_csv(r)=="event_time_us,nozzle_index,command\n1450000,1,ON\n1450000,2,ON\n1550000,1,OFF\n1550000,2,OFF\n","CSV actual log");
    off(r);
}
void visuals() {
    for(auto r:{run(),run({detection(1,0)}),run({detection(1,1),detection(2,1,640,1000)})}) {
        const cv::Mat original(1024,2048,CV_8UC3,cv::Scalar(12,34,56)); const auto before=original.clone();
        auto a=demo::render_annotated(original,r), b=demo::render_annotated(original,r);
        check(cv::norm(a,b,cv::NORM_INF)==0,"Annotation pixels deterministic");
        check(cv::norm(original,before,cv::NORM_INF)==0,"Original pixels unchanged");
        a=demo::render_timeline(r); b=demo::render_timeline(r);
        check(a.rows==880 && cv::norm(a,b,cv::NORM_INF)==0,"Timeline pixels deterministic");
    }
}
void output_and_hashes() {
    // Unique temporary location reserved atomically; never delete a pre-existing directory.
    auto root=std::filesystem::temp_directory_path()/"smartspray-demo-tests";
    unsigned suffix=0;
    while(!std::filesystem::create_directory(root)) root=std::filesystem::temp_directory_path()/("smartspray-demo-tests-"+std::to_string(++suffix));
    struct Cleanup {std::filesystem::path p; ~Cleanup(){std::error_code e;std::filesystem::remove_all(p,e);}} cleanup{root};
    const auto file=root/"hash.txt";
    const auto hash=[&](const std::string& content) {
        {std::ofstream f(file,std::ios::binary); f<<content;}
        return sha256_file(file.string());
    };
    check(hash("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty");
    check(hash("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 abc");
    check(hash(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA256 multi-block");
    fails([&]{sha256_file((root/"missing").string());});
    auto r=run({detection(1,1)});
    auto a=demo::render_annotated(cv::Mat(1024,2048,CV_8UC3,cv::Scalar(0,0,0)),r);
    auto t=demo::render_timeline(r); const auto out=root/"run";
    demo::write_outputs(out,r,a,t);
    check(std::distance(std::filesystem::directory_iterator(out),std::filesystem::directory_iterator{})==4,"Complete four-file set");
    std::ifstream in(out/"run.json"); json saved; in>>saved; check(saved==r,"Saved run");
    check(cv::norm(cv::imread((out/"annotated.png").string()),a,cv::NORM_INF)==0,"PNG pixels");
    const auto before=sha256_file((out/"run.json").string());
    fails([&]{demo::write_outputs(out,r,a,t);});
    check(before==sha256_file((out/"run.json").string()),"Refused overwrite preserves run");
    fails([&]{demo::write_outputs(root/"absent"/"run",r,a,t);});
    fails([&]{demo::write_outputs(root/"bad-render",r,cv::Mat(),t);});
    check(!std::filesystem::exists(root/"bad-render"),"No partial encode output");
    std::filesystem::create_symlink(out,root/"alias"); fails([&]{demo::write_outputs(root/"alias",r,a,t);});
    std::filesystem::create_symlink(root/"missing",root/"dangling"); fails([&]{demo::write_outputs(root/"dangling",r,a,t);});
}
}
int main() {
    const std::pair<const char*,void(*)()> tests[]={{"weed_only_selection_and_original_center",selection_mapping},
        {"all_channel_boundaries",all_boundaries},{"image_domain_limits",image_limits},
        {"stable_ids_touch_overlap_attribution",stable_ids_merging},{"empty",empty},{"crops_only",crops_only},
        {"all_rejected",all_rejected},{"mixed",mixed},{"deadline_equality_and_one_us_miss",deadline},
        {"int64_limits",time_limits},{"invalid_configuration",config_validation},
        {"invalid_detections",invalid_detection},{"json_csv_executor_consistency",json_csv},
        {"render_repeat_and_source_preservation",visuals},{"output_transaction_and_sha256",output_and_hashes}};
    unsigned passed=0;
    for(const auto& test:tests) {
        try {test.second(); ++passed; std::cout<<"PASS "<<test.first<<'\n';}
        catch(const std::exception& e) {std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<'\n';return 1;}
    }
    std::cout<<passed<<" demo integration scenarios passed\n";
}
