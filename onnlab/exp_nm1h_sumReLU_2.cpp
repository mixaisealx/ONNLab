#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "OptimizerGD.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_ReLU.h"
#include "NNB_LinearSlim.h"
#include "NNB_ConstInput.h"
#include "NNB_nm1ReLU.h"
#include "LearnGuiderFwBPg.h"
#include "Monotonic2FieldsProjectingAccessory.h"
#include "Monotonic2FieldsHeuristicsEqExV1.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>

#include <iostream>

void exp_nm1h_sumReLU_2() {
	std::cout << "exp_nm1h_sumReLU_2" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

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

	// m1ReLU variant out
	const float NM1_MAX_VALUE = 1.0f;

	nn::NNB_nm1ReLU m1out(NM1_MAX_VALUE, 0.1f);

	// Connections
	const float LEARNING_RATE = 0.1f;
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD(LEARNING_RATE);

	nn::NNB_Connection<OptimGD> connections_stand[] = {
		nn::NNB_Connection<OptimGD>(&bias, &ninc, &optimizerGD), // Bias first, it's important
		nn::NNB_Connection<OptimGD>(&bias, &ndec, &optimizerGD), // Bias first, it's important
		nn::NNB_Connection<OptimGD>(&in1, &ninc, &optimizerGD), // Input order is imoprtant too!
		nn::NNB_Connection<OptimGD>(&in1, &ndec, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &ninc, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &ndec, &optimizerGD)
	};
	nn::NNB_StraightConnection connections_const[] = {
		nn::NNB_StraightConnection(&ninc, &out),
		nn::NNB_StraightConnection(&ndec, &out)
	};

	nn::NNB_Connection<OptimGD> connections_m1[] = {
		nn::NNB_Connection<OptimGD>(&bias, &m1out, &optimizerGD), // Bias first, it's important
		nn::NNB_Connection<OptimGD>(&in1, &m1out, &optimizerGD), // Input order is imoprtant too!
		nn::NNB_Connection<OptimGD>(&in2, &m1out, &optimizerGD)
	};

	// Perfect result storage
	std::vector<float> perfect_out({ 0 });

	nn::NNB_Layer stand_layer1({ &ninc, &ndec });
	nn::NNB_Layer stand_layer2({ &out });

	nn::NNB_Layer m1_layer1({ &m1out });

	// Initializing connections
	for (auto &conn : connections_stand) {
		conn.Weight(randistributor(preudorandom));
	}

	for (auto &conn : connections_m1) {
		conn.Weight(randistributor(preudorandom)); // Non-optimal init
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
	
	nn::Monotonic2FieldsHeuristicsEqExV1 mon2filds(NM1_MAX_VALUE);
	nn::Monotonic2FieldsProjectingAccessory stand_m1_proj(&m1_layer1, &stand_layer1, &mon2filds);

	nn::errcalc::ErrorCalcMSE mse_calculator(perfect_out.size());

	nn::LearnGuiderFwBPg stand_learnguider({ &stand_layer1, &stand_layer2 }, &mse_calculator);
	nn::LearnGuiderFwBPg m1_learnguider({ &m1_layer1 }, &mse_calculator);

	bool stand_learn = true;

	std::uniform_int_distribution<unsigned> testselector(0, 3);
	for (size_t iterations = 0; iterations < 100; ++iterations) {
		// Select datarow
		const auto &row = traindata[testselector(preudorandom)];

		// Update inputs
		inputs_store[0] = row.input[0];
		inputs_store[1] = row.input[1];
		perfect_out[0] = row.output;

		// Learn
		if (stand_learn) {
			stand_learnguider.DoForward();
			stand_learnguider.FillupOutsError(perfect_out);
			stand_learnguider.DoBackward();
		} else {
			m1_learnguider.DoForward();
			m1_learnguider.FillupOutsError(perfect_out);
			m1_learnguider.DoBackward();
		}

		// Knowledge transfer
		if (!((iterations + 1) % 25)) {
			if (stand_learn) {
				stand_m1_proj.Perform2to1LossyCompression();
			} else {
				stand_m1_proj.Perform1to2DiffTransfer();
			}
			stand_learn = !stand_learn;
		}
	}

	std::vector<std::tuple<float, float>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		inputs_store[0] = sample.input[0];
		inputs_store[1] = sample.input[1];

		// Output layer forward
		m1_learnguider.DoForward();

		results.push_back(std::make_tuple(m1out.OwnLevel(), sample.output));
	}

	for (const auto &tpl : results) {
		std::cout << std::get<1>(tpl) << '\t' << std::get<0>(tpl) << std::endl;
	}

	return;
}