#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <fstream>
#include <unordered_map>
#include <cctype>
#include <cmath>

/**
 * multinomial naive bayse
 * normal bayse but with a series
 */


using SMSFeatureField_t = std::unordered_map<std::string,int>;

struct SMSData{
    int spam_count{0};
    int ham_count{0};
    int total_spam_words{0};
    int total_ham_words{0};
    size_t vocab_size{0};
    SMSFeatureField_t spam_feature_count;
    SMSFeatureField_t ham_feature_count;
    std::unordered_set<std::string> vocab;
};

// clear the punctuation and normalise the spacing
char sanitiseLine(char c)
{
    if (std::isalnum(static_cast<unsigned char>(c))) {
        return std::tolower(static_cast<unsigned char>(c));
    }
    return ' ';
}


void parseLine(const std::string& line, SMSData *data){
    std::istringstream iss{line};
    std::string str;
    std::string clss;
    iss >> clss;

   SMSFeatureField_t *features;
   int *totalWords;
   // get the  feature count per line and total count
   if(clss == "spam"){
       data->spam_count++;
       features = &data->spam_feature_count;
       totalWords = &data->total_spam_words;
   }else if(clss == "ham"){
       data->ham_count++;
       features = &data->ham_feature_count;
       totalWords = &data->total_ham_words;
   }else{
       return;
   }

   while(iss >> str){
       (*features)[str]++;
       (*totalWords)++;
       data->vocab.insert(str);
   }
}

void parseCSVRow(std::string_view line, SMSData *data){
    std::string clean_line;
    for(auto val : line){
        clean_line.push_back(sanitiseLine(val));
    }
    parseLine(clean_line, data);
}

// read the dataset and
void handle_dataset(std::string filepath, SMSData *data){
    std::ifstream file{filepath.data()};
    if (!file) {
        std::cerr << "Failed to open: " << filepath << '\n';
        return;
    }
    std::string row;
    while(std::getline(file, row)){
        parseCSVRow(row, data);
    }
    data->vocab_size = data->vocab.size();
}

double p_word_given_spam(const std::string& word,const SMSData& data){
    size_t count = 0;

    if (auto it = data.spam_feature_count.find(word);
        it != data.spam_feature_count.end())
    {
        count = it->second;
    }

    return static_cast<double>(count + 1)
        / (data.total_spam_words + data.vocab_size);
}

double p_word_given_ham(const std::string& word,const SMSData& data){
    size_t count = 0;

    if (auto it = data.ham_feature_count.find(word);
        it != data.ham_feature_count.end())
    {
        count = it->second;
    }

    // laplace smoothing
    return static_cast<double>(count + 1)
        / (data.total_ham_words + data.vocab_size);
}

bool predict(
    std::string_view message,
    const std::unordered_map<std::string, double>& spam_weights,
    const std::unordered_map<std::string, double>& ham_weights,
    double log_prior_spam,
    double log_prior_ham,
    double log_unknown_spam,
    double log_unknown_ham)
{
    // Start with priors
    double spam_score = log_prior_spam;
    double ham_score  = log_prior_ham;

    // Clean message
    std::string clean;
    for(char c : message) {
        clean.push_back(sanitiseLine(c));
    }

    // Tokenize
    std::istringstream iss{clean};
    std::string word;

    while(iss >> word) {
        if(auto it = spam_weights.find(word);
           it != spam_weights.end())
        {
            spam_score += it->second;
        }
        else
        {
            spam_score += log_unknown_spam;
        }

        if(auto it = ham_weights.find(word);
           it != ham_weights.end())
        {
            ham_score += it->second;
        }
        else
        {
            ham_score += log_unknown_ham;
        }
    }

    std::cout << "Spam log-score: " << spam_score << '\n';
    std::cout << "Ham log-score : " << ham_score << '\n';

    std::cout << ((spam_score < ham_score) ? "It's HAM!\n" : "It's Spam!\n");

    return spam_score > ham_score;
}

int main(){
    SMSData data;
    handle_dataset("dataset/SMSSpamCollection.tsv", &data);
    std::cout <<"spam_count: "<< data.spam_count << '\n';
    std::cout <<"ham_count: " << data.ham_count << '\n';

    double p_spam{(double)data.spam_count/(data.spam_count+data.ham_count)};
    double p_ham{(double)data.ham_count/(data.spam_count+data.ham_count)};

    std::unordered_map<std::string, double> spam_weights;
    std::unordered_map<std::string, double> ham_weights;

    for(const auto& word : data.vocab){
        spam_weights[word] =
            std::log(p_word_given_spam(word, data));

        ham_weights[word] =
            std::log(p_word_given_ham(word, data));
    }

    double log_prior_spam = std::log(p_spam);
    double log_prior_ham  = std::log(p_ham);

   auto log_unknown_spam =
       std::log(
           1.0 /
           (data.total_spam_words + data.vocab_size)
   );

   auto log_unknown_ham =
       std::log(
           1.0 /
           (data.total_ham_words + data.vocab_size)
   );

   std::string line;
   std::cout << "Enter line: ";
   std::getline(std::cin, line);
   predict(
       line,
       spam_weights,
       ham_weights,
       log_prior_spam,
       log_prior_ham,
       log_unknown_spam,
       log_unknown_ham);

    return 0;
}
