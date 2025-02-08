#include "onnlab.h"

#include <iostream>

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "NNB_ConnHyperGraphAggregator.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_m1h_nanReLU.h"
#include "NNB_m1h_SelectorHead.h"
#include "NNB_ConstInput.h"
#include "FwdBackPropGuider.h"

#include <random>
#include <vector>
#include <tuple>
#include <algorithm>


void exp_m1hReLU_1() {
	std::cout << "exp_m1hReLU_1" << std::endl;

	std::random_device randevice;
	std::mt19937 preudorandom(randevice());
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	float inp[2] = { 0, 1 };

	// Input layer
	nn::NNB_Input in1(inp), in2(inp + 1);
	nn::NNB_ConstInput bias;

	// Hidden layer
	nn::NNB_m1h_nanReLU<true> ninc;
	nn::NNB_m1h_nanReLU<false> ndec;

	// Output neuron
	nn::NNB_m1h_SelectorHead<false> out;

	// Connections
	nn::NNB_Connection connections[] = {
		nn::NNB_Connection(&in1, &ninc),
		nn::NNB_Connection(&in2, &ninc),
		nn::NNB_Connection(&in1, &ndec),
		nn::NNB_Connection(&in2, &ndec)
	};
	nn::NNB_Connection connections_hypergraph[] = {
		nn::NNB_Connection(&bias, &ninc),
		nn::NNB_Connection(&bias, &ndec)
	};
	nn::NNB_StraightConnection connections_str[] = {
		nn::NNB_StraightConnection(&ninc, &out),
		nn::NNB_StraightConnection(&ndec, &out)
	};

	nn::NNB_ConnHyperGraphAggregator bias_hpg({ &connections_hypergraph[0], &connections_hypergraph[1] });

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer layer1({ &ninc, &ndec });
	nn::NNB_Layer layer2({ &out });

	// Initialising connections
	for (auto &conn : connections) {
		conn.Weight(randistributor(preudorandom));
	}
	// Initialising bias
	for (auto &conn : connections_hypergraph) {
		//conn.Weight(randistributor(preudorandom));
		conn.Weight(-1.0f);
	}
	bias_hpg.DoWeightsProcessing();

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

	nn::BasicOutsErrorSetter::ErrorCalcMSE mse_calculator;
	nn::BasicOutsErrorSetter nerr_setter(&mse_calculator, perfect_out.size());
	nn::FwdBackPropGuider learnguider({ &layer1, &layer2 }, &nerr_setter);

	std::uniform_int_distribution<unsigned> testselector(0, 3);
	for (size_t iterations = 0; iterations < 1000; ++iterations) {
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
		bias_hpg.DoWeightsProcessing();

		if ((iterations + 1) % 10 == 0) {
			for (auto &itm : traindata) { // Adjust the weights over dataset so there is only one output value (not nan).
				inp[0] = itm.input[0];
				inp[1] = itm.input[1];
				learnguider.DoForward();
				out.NormalizeOwnLevel();
				bias_hpg.DoWeightsProcessing();
			}
		}
	}

	// Final weight correction
	/*for (auto &itm : traindata) {
		inp[0] = itm.input[0];
		inp[1] = itm.input[1];
		learnguider.DoForward();
		out.NormalizeOwnLevel();
		bias_hpg.DoWeightsProcessing();
	}*/


	std::vector<std::vector<float>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		inp[0] = sample.input[0];
		inp[1] = sample.input[1];

		// Output layer forward
		learnguider.DoForward();

		if (std::isnan(out.OwnLevel())) {
			results.push_back({ sample.output });
			for (auto &itm : out.RetriveCandidates()) {
				results.back().push_back(itm.value);
			}
		} else {
			results.push_back({ sample.output,  out.OwnLevel() });
		}
	}

	for (const auto &tpl : results) {
		for (auto itm : tpl) {
			std::cout << itm << '\t';
		}
		std::cout << std::endl;
	}

	return;
}