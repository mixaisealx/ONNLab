#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerGD.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_nm1ReLU.h"
#include "NNB_ConstInput.h"
#include "LearnGuiderFwBPg.h"

#include <random>
#include <vector>
#include <tuple>

#include <iostream>


void exp_nm1ReLU_5() {
	std::cout << "exp_nm1ReLU_5" << std::endl;

	std::random_device randevice;
	//std::mt19937 preudorandom(42);
	auto randval = randevice();
	std::cout << randval << std::endl;
	std::mt19937 preudorandom(randval);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	float inp[2] = { 0, 1 };

	// Input layer
	nn::NNB_Input in1(inp), in2(inp + 1);
	nn::NNB_ConstInput bias;

	// Output neuron
	nn::NNB_nm1ReLU out(1.0f, 0.1f);

	// Connections
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD;

	nn::NNB_Connection<OptimGD> connections[] = {
		nn::NNB_Connection<OptimGD>(&in1, &out, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &out, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&bias, &out, &optimizerGD)
	};

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer layer1({ &out });

	// Initialising biases

	// Initialising connections
	for (auto &conn : connections) {
		conn.Weight(randistributor(preudorandom));
	}

	// Train data
	struct datarow {
		datarow(float in1, float in2, float out) {
			input[0] = in1;
			input[1] = in2;
			output = out;
		}
		float input[2];
		float output;
	};

	std::vector<datarow> traindata = { datarow(0,0,0), datarow(0,1,1), datarow(1,0,1), datarow(1,1,0) };

	nn::errcalc::ErrorCalcMSE mse_calculator(perfect_out.size());
	nn::LearnGuiderFwBPg learnguider({ &layer1 }, &mse_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, 3);

	struct deviation {
		deviation():distance(-1), best_canditate(nullptr) {}
		float distance;
		datarow *best_canditate;
	};
	std::array<deviation, 2> deviation_store;

	for (size_t iterations = 0; iterations < 100; ++iterations) {
		// Select datarow
		const auto &row = traindata[testselector(preudorandom)];
		//const auto &row = traindata[0];

		// Update inputs
		inp[0] = row.input[0];
		inp[1] = row.input[1];
		perfect_out[0] = row.output;

		learnguider.DoForward();
		learnguider.FillupOutsError(perfect_out);
		learnguider.DoBackward();

		if (iterations < 52 && (iterations + 1) % 3 == 0) {
			// Scan dataset for usage
			deviation_store.fill(deviation());
			out.BatchAnalyzer_Reset();
			auto &deviation_curr = out.BatchAnalyzer_GetFieldsActivateMinDistance();

			out.BatchAnalyzer_SetState(true);
			for (auto &itm : traindata) {
				inp[0] = itm.input[0];
				inp[1] = itm.input[1];
				learnguider.DoForward();

				auto first1 = deviation_store.begin();
				auto last1 = deviation_store.end();
				auto first2 = deviation_curr.begin();
				while (first1 != last1) {
					if (first1->distance != *first2) {
						first1->distance = *first2;
						first1->best_canditate = &itm;
					}
					++first1; ++first2;
				}
			}
			out.BatchAnalyzer_SetState(false);

			for (unsigned char idx = 0; idx != 2; ++idx) {
				auto &ref = deviation_store[idx];
				if (ref.distance > 1e-7f && ref.distance < 0.5f) { // The field has not met && distance to another field not so big
					inp[0] = ref.best_canditate->input[0];
					inp[1] = ref.best_canditate->input[1];
					learnguider.DoForward();
					out.FlipCurrentInputToAnotherField(); // Moving current input processing to another part of activaiton function
				}
			}
		}
	}

	std::vector<std::tuple<float, float>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		inp[0] = sample.input[0];
		inp[1] = sample.input[1];

		// Output layer forward
		out.UpdateOwnLevel();

		results.push_back(std::make_tuple(out.OwnLevel(), sample.output));
	}

	for (const auto &tpl : results) {
		std::cout << std::get<1>(tpl) << '\t' << std::get<0>(tpl) << std::endl;
	}

	return;
}