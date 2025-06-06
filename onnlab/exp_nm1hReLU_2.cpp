#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "NNB_ConnWghAverager.h"
#include "OptimizerGD.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_nm1h_nanReLU.h"
#include "NNB_nm1h_SelectorHead.h"
#include "NNB_ConstInput.h"
#include "LearnGuiderFwBPg.h"

#include <random>
#include <vector>
#include <tuple>
#include <algorithm>

#include <iostream>


void exp_nm1hReLU_2() {
	std::cout << "exp_nm1hReLU_2" << std::endl;

	std::random_device randevice;
	std::mt19937 preudorandom(randevice());
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	float inp[2] = { 0, 1 };

	// Input layer
	nn::NNB_Input in1(inp), in2(inp + 1);
	nn::NNB_ConstInput bias;

	// Hidden layer
	nn::NNB_nm1h_nanReLU<true> ninc;
	nn::NNB_nm1h_nanReLU<false> ndec;

	// Output neuron
	nn::NNB_nm1h_SelectorHead<true> out;

	// Connections
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD;

	nn::NNB_Connection<OptimGD> connections[] = {
		nn::NNB_Connection<OptimGD>(&in1, &ninc, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in1, &ndec, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &ninc, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &ndec, &optimizerGD)
	};
	nn::NNB_Connection<OptimGD> connections_hypergraph[] = {
		nn::NNB_Connection<OptimGD>(&bias, &ninc, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&bias, &ndec, &optimizerGD)
	};
	nn::NNB_StraightConnection connections_str[] = {
		nn::NNB_StraightConnection(&ninc, &out),
		nn::NNB_StraightConnection(&ndec, &out)
	};

	nn::NNB_ConnWghAverager bias_hpg({ &connections_hypergraph[0], &connections_hypergraph[1] });

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
		conn.Weight(-0.2f);
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

	nn::errcalc::ErrorCalcMSE mse_calculator(perfect_out.size());
	nn::LearnGuiderFwBPg learnguider({ &layer1, &layer2 }, &mse_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, 3);
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
		bias_hpg.DoWeightsProcessing();
	}

	std::vector<std::tuple<float, float>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		inp[0] = sample.input[0];
		inp[1] = sample.input[1];

		// Output layer forward
		learnguider.DoForward();

		results.push_back(std::make_tuple(out.OwnLevel(), sample.output));
	}

	for (const auto &tpl : results) {
		std::cout << std::get<1>(tpl) << '\t' << std::get<0>(tpl) << std::endl;
	}

	return;
}