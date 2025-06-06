#include "onnlab.h"

#include "NNB_Connection.h"
#include "OptimizerGD.h"
#include "NNB_iReLU.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "LearnGuiderFwBPg.h"

#include <random>
#include <array>
#include <vector>
#include <tuple>

#include <iostream>


void exp_iReLU_1() {
	std::cout << "exp_iReLU_1" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);

	// Input vector
	std::array<float, 2> inputs_store;

	// Input layer
	nn::NNB_Input in1(&inputs_store[0]), in2(&inputs_store[1]);

	// Hidden layer
	nn::NNB_iReLU lr11(0.1f), lr12(0.1f);
	// Output neuron
	nn::NNB_iReLU out(0.1f);

	const float LEARNING_RATE = 0.1f;

	// Connections
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD(LEARNING_RATE);

	nn::NNB_Connection<OptimGD> connections[] = {
		nn::NNB_Connection<OptimGD>(&in1, &lr11, &optimizerGD, 1.0f),
		nn::NNB_Connection<OptimGD>(&in1, &lr12, &optimizerGD, 1.0f),
		nn::NNB_Connection<OptimGD>(&in2, &lr11, &optimizerGD, 1.0f),
		nn::NNB_Connection<OptimGD>(&in2, &lr12, &optimizerGD, 1.0f),
		nn::NNB_Connection<OptimGD>(&lr11, &out, &optimizerGD, -0.2f),
		nn::NNB_Connection<OptimGD>(&lr12, &out, &optimizerGD, -0.2f)
	};

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer layer1({ &lr11, &lr12 });
	nn::NNB_Layer layer2({ &out });

	// Initialising connections
	for (auto &conn : connections) {
		conn.Weight(conn.Weight() * randistributor(preudorandom));
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
	for (size_t iterations = 0; iterations < 250; ++iterations) {
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

		learnguider.DoForward();

		results.push_back(std::make_tuple(out.OwnLevel(), sample.output));
	}

	for (const auto &tpl : results) {
		std::cout << std::get<1>(tpl) << '\t' << std::get<0>(tpl) << std::endl;
	}

	return;
}