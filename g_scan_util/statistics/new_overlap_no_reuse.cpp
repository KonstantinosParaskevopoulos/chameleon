#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <cmath>
#define SMALLEST_REGION_THRESHOLD 10 //Region below x Number of Scores is not detected
#define ACCEPTABLE_SUCCESS_LIMIT 3 //A detection must be within x number of scan windows to be considered successful 
//#define NDEBUG

using namespace std;

struct region
{
    int start;
    int end;
};

struct classifiedWindow
{
    int target;
    // int     window_size;
    float bin_class;
};

struct Th_classifiedWindow
{
    int target;
    // int     window_size;
    bool bin_class;
};

struct Match
{
    int gt_idx;
    int det_idx;
    float iou;
};

struct RegionMetrics
{
    int TP = 0, FP = 0, FN = 0;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;
    float meanIoU = 0.0f;
};

struct BoundaryMetrics
{
    float mean_start_error = 0.0f;
    float mean_end_error   = 0.0f;
    float mean_total_error = 0.0f;
    float mean_error_comparison =0.0f;
    float start_success_percent = 0.0f;             // Number of Successful Detections accross Detections
    float start_success_percent_overall = 0.0f;     // Number of Successful Detections accross All GT Regions
    float end_success_percent = 0.0f;               // Number of Successful Detections accross Detections
    float end_success_percent_overall = 0.0f;       // Number of Successful Detections accross All GT Regions
    int max_error = 0;
};


 // ======================= IoU =======================
float computeIoU(const region& a, const region& b)
{
    int intersection_start = max(a.start, b.start);
    int intersection_end   = min(a.end, b.end);

    int intersection = max(0, intersection_end - intersection_start);
    int union_val = max(a.end, b.end) - min(a.start, b.start);

    if (union_val == 0) return 0.0f;

    return (float)intersection / union_val;
}


// ======================= MATCHING =======================
vector<Match> matchRegions(
    const vector<region>& detected,
    const vector<region>& truth,
    float iou_threshold)
{
    vector<Match> matches;
    vector<bool> used(truth.size(), false); //Was Commented

    for (size_t i = 0; i < detected.size(); i++)
    {
        float best_iou = 0.0f;
        int best_idx = -1;

        for (size_t j = 0; j < truth.size(); j++)
        {
            if (used[j]) continue;  //Was Commented

            float iou = computeIoU(detected[i], truth[j]);

            if (iou >= best_iou)
            {
                best_iou = iou;
                best_idx = j;
            }
            
        }

        //if (best_iou >= iou_threshold)  //IoU threshold is the value below which a region is not ignored, normally 0.001f as with 0 -> Core Dump
        if (best_idx != -1 && best_iou >= iou_threshold)
        {
            matches.push_back({best_idx, (int)i, best_iou});
            used[best_idx] = true;  //Was Commented
        }
    }

    return matches;
}


// ======================= REGION METRICS =======================
RegionMetrics computeRegionMetrics(
    const vector<region>& detected,
    const vector<region>& truth,
    float iou_threshold)
{
    RegionMetrics m;

    auto matches = matchRegions(detected, truth, iou_threshold);
    // FN: truth regions not overlapped by ANY detection
    vector<bool> covered(truth.size(), false);


    m.TP = matches.size();
    m.FP = detected.size() - m.TP;
    
    // if (detected.size()>truth.size()){
    //     m.FN = truth.size() - m.TP;
    // }
    //else {
    for (const auto& match : matches){
        covered[match.gt_idx] = true;
    }
    m.FN = 0;

    for (bool c : covered){
        if (!c) m.FN++;
    }
    //}
       

    float sum_iou = 0.0f;
    for (const auto& match : matches)
        sum_iou += match.iou;

    m.meanIoU = (m.TP > 0) ? sum_iou / m.TP : 0.0f;
    
    if (m.TP + m.FP > 0)
        m.precision = (float)m.TP / (m.TP + m.FP);

    if (m.TP + m.FN > 0)
        m.recall = (float)m.TP / (m.TP + m.FN);

    if (m.precision + m.recall > 0)
        m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall);

    return m;
}


// ======================= BOUNDARY METRICS =======================
BoundaryMetrics computeBoundaryMetrics(
    const vector<region>& detected,
    const vector<region>& truth,
    const vector<Match>& matches, int window_size)
{
    BoundaryMetrics m;

    if (matches.empty()) return m;

    int sum_start = 0;
    int sum_end   = 0;
    int sum_total = 0;
    int max_err   = 0;
    
    int sum_successful_start = 0;
    int sum_successful_end = 0;

    int sum_enc_lengths = 0;
    int average_enc_length = 0;

    for (const auto& match : matches)
    {
        const region& d = detected[match.det_idx];
        const region& t = truth[match.gt_idx];
        sum_enc_lengths += t.end-t.start;
        int start_err = abs(d.start - t.start);
        int end_err   = abs(d.end   - t.end);
        int total_err = (int)((start_err + end_err)/2); //Average of start & end error
        //int total_err = start_err + end_err; //Total start+end error
        sum_start += start_err;
        sum_end   += end_err;
        sum_total += total_err;

        max_err = max(max_err, max(start_err, end_err));

        if(start_err<=(ACCEPTABLE_SUCCESS_LIMIT*window_size)/2){
            sum_successful_start++;
        }
        if(end_err<=(ACCEPTABLE_SUCCESS_LIMIT*window_size)/2){
            sum_successful_end++;
        }
    }

    int NO_MATCHES = detected.size();
    int NO_TRUTH_REGIONS = truth.size();
    average_enc_length = (int)(sum_enc_lengths / NO_TRUTH_REGIONS);

    m.mean_start_error = (float)sum_start / NO_MATCHES;
    m.mean_end_error   = (float)sum_end   / NO_MATCHES;
    m.mean_total_error = (float)sum_total / NO_MATCHES;
    m.max_error        = max_err;
    m.start_success_percent = (float)sum_successful_start / NO_MATCHES;
    m.end_success_percent = (float)sum_successful_end / NO_MATCHES;
    m.start_success_percent_overall = (float)sum_successful_start / NO_TRUTH_REGIONS;
    m.end_success_percent_overall = (float)sum_successful_end / NO_TRUTH_REGIONS;
    m.mean_error_comparison = (float)m.mean_total_error/average_enc_length;
    return m;
}


// ======================= FINAL EVALUATION =======================
void evaluateAESRegions(
    const vector<region>& detected,
    const vector<region>& truth,
    float iou_threshold, float threshold_og, int filter_size, int window_size)
{
    auto matches = matchRegions(detected, truth, iou_threshold);

    auto regionMetrics = computeRegionMetrics(detected, truth, iou_threshold);
    auto boundaryMetrics = computeBoundaryMetrics(detected, truth, matches, window_size);
    
    string iou_str = to_string((int)(threshold_og*100));
    string f_str = to_string(filter_size);
    string filename = "results_thres_0_" + iou_str + "_filt_" + f_str +".txt";
    //cout<<filename<<endl;
    ofstream outFile(filename);
    //ofstream outFile("report.txt");

    if (!outFile.is_open()) {
        std::cerr << "Error opening file!\n";
        return;
    }

    outFile << "===== REGION DETECTION =====\n";
    outFile << "TP: " << regionMetrics.TP << "\n";
    outFile << "FP: " << regionMetrics.FP << "\n";
    outFile << "FN: " << regionMetrics.FN << "\n";
    //cout << "Precision: " << regionMetrics.precision << "\n";
    //cout << "Recall: " << regionMetrics.recall << "\n";
    //cout << "F1 Score: " << regionMetrics.f1 << "\n";
    float IoU_perc = regionMetrics.meanIoU * 100;
    outFile << "Mean IoU: " << IoU_perc << "%\n\n";
    cout << static_cast<long long>(boundaryMetrics.mean_total_error) << "\t";//regionMetrics.meanIoU << "\t"; //<< filter_size << "\t" << threshold_og <<endl;
    outFile << "===== BOUNDARY ACCURACY =====\n";
    outFile << "Mean Start Error: " << static_cast<long long>(boundaryMetrics.mean_start_error) << "\n";
    outFile << "Mean End Error: " << static_cast<long long>(boundaryMetrics.mean_end_error) << "\n";
    outFile << "Mean Total Error: " << static_cast<long long>(boundaryMetrics.mean_total_error) << "\n";
    outFile << "Mean Error (Perc. of Enc.): " << boundaryMetrics.mean_error_comparison*100 << "%\n\n";
    //outFile << "Max Error: " << boundaryMetrics.max_error << "\n\n";
    outFile << "===== DETENTION ACCURACY =====\n";
    outFile << "Successfull Start Detections:" << (boundaryMetrics.start_success_percent)*100 << "%\n";
    outFile << "Successfull Start Detections (OVR):" << (boundaryMetrics.start_success_percent_overall)*100 << "%\n";
    outFile << "Successfull End Detections:" << (boundaryMetrics.end_success_percent)*100 << "%\n";
    outFile << "Successfull End Detections (OVR):" << (boundaryMetrics.end_success_percent_overall)*100 << "%\n";
    outFile << "Successfull Detections:" << ((boundaryMetrics.start_success_percent + boundaryMetrics.end_success_percent)/2)*100 << "%\n";
    outFile << "Successfull Detections (Overall):" << ((boundaryMetrics.start_success_percent_overall + boundaryMetrics.end_success_percent_overall)/2)*100 << "%\n";

    outFile.close();
}


/******************************************************************************************/


int ReadRegionFile(ifstream &fin, vector<region> &AES_regions_array)
{
    /* Function to read Pinpoints TXT */
    int i = 0;
    if (!fin)
    {
        cout << "Pinpoint File Corrupted or doesn't Exist!" << endl;
        return 0;
    }
    region temp;
    while (fin >> temp.start >> temp.end)
    {
        assert(temp.end > temp.start);
        AES_regions_array.push_back(temp);
        i++;
    }
    return i;
}

int ReadReportFile(ifstream &fin, vector<classifiedWindow> &data)
{
    /* Function to read RAiSD-AI Report */
    if (!fin)
    {
        cout << "Report File Corrupted or doesn't Exist!" << endl;
        return 0;
    }
    float dummy[7];
    int i = 0;
    classifiedWindow temp;
    while (fin >> temp.target >> dummy[0] >> dummy[1] >>dummy[2] >>dummy[3] >>dummy[4] >>dummy[5] >> temp.bin_class >>dummy[6])
    {   //l in with temp values and read only first and second to last columns
        //Threshold values
        data.push_back(temp);
        i++;
    }

    return i; // number of records read
}

vector<classifiedWindow> SlidingWindowAverage(vector<classifiedWindow> &data, int win_size){
    float temp_sum;
    float averaged_val;

    assert(win_size > 0);
    assert(win_size <= data.size());
    vector<classifiedWindow> OutputVector;
    for(int i=0; i<=(data.size()-win_size); i++)
    {
        temp_sum=0.0f;
        for (int j = i; j < i + win_size; j++)
        {
            temp_sum += data.at(j).bin_class;
        }
        averaged_val = temp_sum / win_size;
        classifiedWindow temp;
        temp.target = data.at(i).target;
        temp.bin_class = averaged_val;
        OutputVector.push_back(temp);
    }
    return OutputVector;
}

vector<Th_classifiedWindow> ThresholdValues(vector<classifiedWindow> &data, float threshold){
    float temp_sum;
    float averaged_val;
    vector<Th_classifiedWindow> OutputVector;
    for(int i=0; i<data.size(); i++)
    {
        Th_classifiedWindow temp;
        temp.target=data.at(i).target;
        if(data.at(i).bin_class<threshold){
            temp.bin_class = (bool)0;
        }
        else{
            temp.bin_class = (bool)1;
        }
        OutputVector.push_back(temp);
    }
    return OutputVector;
}

vector<region> EdgeDetector(const vector<Th_classifiedWindow>& data)
{
    vector<region> detected_regions;

    //if (data.empty()) return detected_regions;

    bool in_region = false;
    region current;

    // Handle initial HIGH
    if (data[0].bin_class == 1)
    {
        current.start = data[0].target;
        in_region = true;
    }

    for (int i = 1; i < data.size(); i++)
    {
        // Rising edge
        if (!in_region && data[i-1].bin_class == 0 && data[i].bin_class == 1)
        {
            current.start = data[i].target;
            in_region = true;
        }

        // Falling edge
        else if (in_region && data[i-1].bin_class == 1 && data[i].bin_class == 0)
        {
            current.end = data[i-1].target;
            if((current.end-current.start) >= SMALLEST_REGION_THRESHOLD){            
                detected_regions.push_back(current);
            }
            in_region = false;
        }
    }

    // Handle signal ending HIGH
    if (in_region)
    {
        current.end = data.back().target;
        detected_regions.push_back(current);
    }

    return detected_regions;
}
void printRegions(vector<region> regionToPrint, float iou_threshold, int filter_size){
    
    string iou_str = to_string((int)(iou_threshold*100));
    string f_str = to_string(filter_size);
    string filename = "detected_regions_thres_0_" + iou_str + "_filt_" + f_str +".txt";
    //cout<<filename<<endl;
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file!\n";
        return;
    }

    for (int i = 0; i<regionToPrint.size(); i++){
        outFile << "Start : "<<regionToPrint[i].start<< " End: "<<regionToPrint[i].end<<endl;
    }
    outFile.close();
}

//ADD check function
int main(int argc, char* argv[])
{
    if (argc != 6) {
        cout << "Usage: " << argv[0] << " <filter_size> <threshold> <report_file> <pinpoint_file> <window_size>" << endl;
        return 1;
    }
    vector<classifiedWindow> classified_win_array;
    vector<region> AES_Regions;
    vector<classifiedWindow> filtered_window;
    vector<Th_classifiedWindow> thresholded_window;
    vector<region> DetectedRegions;
    int report_size, region_size;
    

    int filter_size = atoi(argv[1]);     // convert to int
    float threshold = atof(argv[2]);     // convert to float
    
    string report_file = argv[3];
    string pinpoint_file = argv[4]; 
    int window_size = atoi(argv[5]);

    //ifstream rep("/home/konstantinos/Documents/GitHub/Genome-Scanning/SCRIPTS/RESULTS/W_500_S_10/RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1");///home/konstantinos/Documents/GitHub/Genome-Scanning/Downloads/RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1");//("/home/konstantinos/Documents/GitHub/Genome-Scanning/SCRIPTS/RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1");
    //ifstream pin("/home/konstantinos/Documents/GitHub/Genome-Scanning/Datasets/ISO_AES_FROM_1_VCF_x50_3356317-3739461_o0_chameleon_base_chunk_1_trace_0.txt");///home/konstantinos/Documents/GitHub/Genome-Scanning/Downloads/VCF_x50_0-0_o1267_chameleon_rd_chunk_1_trace_0.txt");//("/home/konstantinos/Documents/GitHub/Genome-Scanning/Datasets/100_AES_BASE_VCF_x50_0-16564324_o0_chameleon_base_chunk_1_trace_0.txt");
    ifstream rep(report_file);//("/home/konstantinos/Documents/GitHub/Genome-Scanning/Downloads/RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1");//("/home/konstantinos/Documents/GitHub/Genome-Scanning/SCRIPTS/RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1");
    ifstream pin(pinpoint_file);//("/home/konstantinos/Documents/GitHub/Genome-Scanning/Downloads/VCF_x50_0-0_o1267_chameleon_rd_chunk_1_trace_0.txt");//("/home/konstantinos/Documents/GitHub/Genome-Scanning/Datasets/100_AES_BASE_VCF_x50_0-16564324_o0_chameleon_base_chunk_1_trace_0.txt");

    report_size = ReadReportFile(rep, classified_win_array);
    region_size = ReadRegionFile(pin, AES_Regions);
    rep.close();
    pin.close();

    filtered_window = SlidingWindowAverage(classified_win_array, filter_size);
    // for (int i = 0; i<filtered_window.size(); i++){
    //     cout << "Target : "<<filtered_window[i].target<< " Bin_class: "<<filtered_window[i].bin_class<< " Diff: "<< abs((filtered_window[i].bin_class-classified_win_array[i].bin_class)) <<endl;
    // }
    thresholded_window = ThresholdValues(filtered_window, threshold);
    //for (int i = 0; i<thresholded_window.size(); i++){
    //    cout << "Target : "<<thresholded_window[i].target<< " Bin_class: "<<thresholded_window[i].bin_class<< " Original Value: "<< thresholded_window[i].bin_class <<endl;
    //}
    DetectedRegions = EdgeDetector(thresholded_window);
    
    float IoU_threshold = 0.000f; //Count regions with Overlap > 0.000%

    evaluateAESRegions(DetectedRegions, AES_Regions, IoU_threshold, threshold, filter_size, window_size);
    
    
    printRegions(DetectedRegions, threshold, filter_size);
    //cout << classified_win_array.at(0).target << endl;
    //cout << classified_win_array.at(0).bin_class << endl;
    
    //cout << "Report Size: " << report_size << endl;
    //cout << "Region File Size: " << region_size << endl;

    // for (size_t i = 0; i < classified_win_array.size(); i++)
    // {
    //     cout << "Target: " << classified_win_array[i].target
    //          << " Class: " << classified_win_array[i].bin_class << endl;
    // }

    return 0;
}
