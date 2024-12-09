#include "onnlab.h"

#include <iostream>

#include "NNB_Connection_spyable.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_m1ReLU.h"
#include "NNB_ConstInput.h"
#include "FwdBackPropGuider.h"

#include <random>
#include <vector>
#include <tuple>


void exp_m1ReLU_3() {
	std::cout << "exp_m1ReLU_3" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);

	// Input vector
	float inp[2] = { 0, 1 };

	// Input layer
	nn::NNB_Input in1(inp), in2(inp + 1);
	nn::NNB_ConstInput bias;

	// Output neuron
	nn::NNB_m1ReLU out;

	// Connections
	nn::NNB_Connection connections[] = {
		nn::NNB_Connection(&in1, &out),
		nn::NNB_Connection(&in2, &out),
		nn::NNB_Connection(&bias, &out)
	};

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer layer1({ &out });

	// Initialising biases

	// Initialising connections
	for (auto &conn : connections) {
		if (dynamic_cast<nn::NNB_ConstInput*>(conn.From())) {
			//conn.Weight(randistributor(preudorandom));
			conn.Weight(-1); // Important!
		} else {
			conn.Weight(randistributor(preudorandom));
		}
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

	nn::BasicOutsErrorSetter::ErrorCalcMSE mse_calculator;
	nn::BasicOutsErrorSetter nerr_setter(&mse_calculator, perfect_out.size());
	nn::FwdBackPropGuider learnguider({ &layer1 }, &nerr_setter);

	std::uniform_int_distribution<unsigned> testselector(0, 3);
	for (size_t iterations = 0; iterations < 10; ++iterations) {
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