#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "OptimizerAdam.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_ReLU.h"
#include "NNB_LinearSlim.h"
#include "NNB_ConstInput.h"
#include "LearnGuiderFwBPg.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>

#include <iostream>

void exp_nm1h_sumReLU_1() {
	std::cout << "exp_nm1h_sumReLU_1" << std::endl;

	//std::random_device randevice;
	std::random_device randevice;
	std::mt19937 preudorandom(randevice());
	std::uniform_real_distribution<float> randistributor(-0.2f, 0.2f);

	// Input vector
	std::array<float, 2> inputs_store;

	// Input layer
	nn::NNB_Input in1(&inputs_store[0]), in2(&inputs_store[1]);
	nn::NNB_ConstInput bias;

	// Hidden layer
	nn::NNB_ReLU ninc(0.1f);
	nn::NNB_ReLU ndec(0.1f);

	// Output neuron
	nn::NNB_LinearSlim out;

	// Connections
	const float LEARNING_RATE = 0.1f;
	using OptimAdam = nn::optimizers::Adam;
	OptimAdam optimizerAdam(LEARNING_RATE);


	nn::NNB_Connection<OptimAdam> connections[] = {
		nn::NNB_Connection<OptimAdam>(&in1, &ninc, &optimizerAdam),
		nn::NNB_Connection<OptimAdam>(&in1, &ndec, &optimizerAdam),
		nn::NNB_Connection<OptimAdam>(&in2, &ninc, &optimizerAdam),
		nn::NNB_Connection<OptimAdam>(&in2, &ndec, &optimizerAdam)
	};
	nn::NNB_Connection<OptimAdam> connections_bias[] = {
		nn::NNB_Connection<OptimAdam>(&bias, &ninc, &optimizerAdam),
		nn::NNB_Connection<OptimAdam>(&bias, &ndec, &optimizerAdam)
	};
	nn::NNB_StraightConnection connections_const[] = {
		nn::NNB_StraightConnection(&ninc, &out),
		nn::NNB_StraightConnection(&ndec, &out)
	};

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer layer1({ &ninc, &ndec });
	nn::NNB_Layer layer2({ &out });

	// Initializing connections
	for (auto &conn : connections) {
		conn.Weight(randistributor(preudorandom));
	}
	for (auto &conn : connections_bias) {
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
	nn::LearnGuiderFwBPg learnguider({ &layer1, &layer2 }, &mse_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, 3);
	for (size_t iterations = 0; iterations < 200; ++iterations) {
		// Select datarow
		const auto &row = traindata[testselector(preudorandom)];

		// Update inputs
		inputs_store[0] = row.input[0];
		inputs_store[1] = row.input[1];
		perfect_out[0] = row.output;

		learnguider.DoForward();
		learnguider.FillupOutsError(perfect_out);
		learnguider.DoBackward();
	}

	std::vector<std::tuple<float, float>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		inputs_store[0] = sample.input[0];
		inputs_store[1] = sample.input[1];

		// Output layer forward
		learnguider.DoForward();

		results.push_back(std::make_tuple(out.OwnLevel(), sample.output));
	}

	for (const auto &tpl : results) {
		std::cout << std::get<1>(tpl) << '\t' << std::get<0>(tpl) << std::endl;
	}

	return;
}