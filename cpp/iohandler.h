#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <valarray>
#include <vector>

class IOHandler {
    public:
        static float * load_to_array(std::string path) {
            std::fstream a(path.c_str());
            std::string l;
            int signal_length = 0;
            while (std::getline(a, l)) ++signal_length;
            float* raw = new float[signal_length];
            int idx = 0;

            std::fstream audiof(path.c_str());
            std::string line;
            while (std::getline(audiof, line)) {
                raw[idx] = std::stof(line);
                ++idx;
            }
            return raw;
        } 

        static std::valarray<float> load(std::string path) {

            std::fstream a(path.c_str());
            std::string l;
            int signal_length = 0;
            while (std::getline(a, l)) ++signal_length;

            std::valarray<float> raw(signal_length);
            int idx = 0;

            std::fstream audiof(path.c_str());
            std::string line;
            while (std::getline(audiof, line)) {
                raw[idx] = std::stof(line);
                ++idx;
            }
            return raw;
        };

        static void write(std::valarray<std::valarray<float>> frames, std::string path) {
            std::ofstream out_file;
            std::ostream_iterator<float> out_it (out_file, ",");

            out_file.open (path);
            for (const auto& f : frames) {
                std::copy(std::begin(f), std::end(f), out_it);
                out_file << std::endl;
            }
            out_file.close();
        };

        static void write(std::vector<std::valarray<float>> frames, std::string path) {
            std::ofstream out_file;
            std::ostream_iterator<float> out_it (out_file, ",");

            out_file.open (path);
            for (const auto& f : frames) {
                std::copy(std::begin(f), std::end(f), out_it);
                out_file << std::endl;
            }
            out_file.close();
        };

        static void write(float frames[], int size, std::string path) {
            std::ofstream out_file;

            out_file.open (path);
            for (int i = 0; i < size; i++) {
                out_file << frames[i] << ",";
            }
            out_file << std::endl;
            out_file.close();
        }

        static void write(float** frames, int num_array, int array_size, const char path[]) {
            std::ofstream out_file;

            out_file.open (path);
            for (int i = 0; i < num_array; i++) {
                for (int j = 0; j < array_size; j++) {
                    out_file << frames[i][j] << ",";                    
                }
                out_file << std::endl;
            }
            out_file.close();
        }

};