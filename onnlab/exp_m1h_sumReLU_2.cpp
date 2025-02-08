#include "onnlab.h"

#include <iostream>

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_ReLU.h"
#include "NNB_m1h_SumHead.h"
#include "NNB_ConstInput.h"
#include "NNB_m1ReLU.h"
#include "FwdBackPropGuider.h"
#include "Monotonic2FeildsProjectingAccessory.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>


void exp_m1h_sumReLU_2() {
	std::cout << "exp_m1h_sumReLU_2" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	std::array<float, 2> inputs_store;

	// Input layer
	nn::NNB_Input in1(&inputs_store[0]), in2(&inputs_store[1]);
	nn::NNB_ConstInput bias;

	// Hidden layer
	nn::NNB_ReLU ninc;
	nn::NNB_ReLU ndec;

	// Output neuron
	nn::NNB_m1h_SumHead out;

	// m1ReLU variant out
	const float M1_MAX_VALUE = 1.0f;

	nn::NNB_m1ReLU m1out(0.1f, M1_MAX_VALUE);

	// Connections
	const float LEARNING_RATE = 0.1f;

	nn::NNB_Connection connections_stand[] = {
		nn::NNB_Connection(&bias, &ninc), // Bias first, it's important
		nn::NNB_Connection(&bias, &ndec), // Bias first, it's important
		nn::NNB_Connection(&in1, &ninc), // Input order is imoprtant too!
		nn::NNB_Connection(&in1, &ndec),
		nn::NNB_Connection(&in2, &ninc),
		nn::NNB_Connection(&in2, &ndec)
	};
	nn::NNB_StraightConnection connections_const[] = {
		nn::NNB_StraightConnection(&ninc, &out),
		nn::NNB_StraightConnection(&ndec, &out)
	};

	nn::NNB_Connection connections_m1[] = {
		nn::NNB_Connection(&bias, &m1out), // Bias first, it's important
		nn::NNB_Connection(&in1, &m1out), // Input order is imoprtant too!
		nn::NNB_Connection(&in2, &m1out)
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
	
	static auto proj_spread = [](float w1, float w2)->float {
		if (w1 > w2) {
			return (w1 - w2) * ((static_cast<int8_t>(w1 * w2 > 0) << 1) - 1);
		} else {
			return (w2 - w1) * ((static_cast<int8_t>(w1 * w2 > 0) << 1) - 1);
		}
	};
	auto proj2to1 = [&](nn::interfaces::NBI *const m1, const nn::interfaces::NBI *const eq1, const nn::interfaces::NBI *const eq2) {
		auto &eq1conn = const_cast<nn::interfaces::NBI *>(eq1)->InputConnections();
		auto &eq2conn = const_cast<nn::interfaces::NBI *>(eq2)->InputConnections();
		auto &m1conn = m1->InputConnections();

		float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
		bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better

		m1conn[0]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f - M1_MAX_VALUE); // Setting [correcting] bias offset
		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = eq1conn[i]->Weight();
			w2 = eq2conn[i]->Weight();
			if (proj_spread(w1, w2) > -1e-3f) { // Same signs (or very small module), suppose single feild usage
				m1conn[i]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
			} else { // Different signs, suppose possible both feilds usage
				// Turning into positive to direct to illinear part (instead of just linear)
				m1conn[i]->Weight((std::fabs(w1) + std::fabs(w2)) / 2.0f); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
			}
		}
	};

	auto proj1to2 = [&](const nn::interfaces::NBI *const m1, nn::interfaces::NBI *const eq1, nn::interfaces::NBI *const eq2) {
		auto &eq1conn = eq1->InputConnections();
		auto &eq2conn = eq2->InputConnections();
		auto &m1conn = const_cast<nn::interfaces::NBI *>(m1)->InputConnections();

		std::vector<std::tuple<float, bool, bool>> m1proj;
		m1proj.reserve(m1conn.size());

		float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
		bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better
		m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f - M1_MAX_VALUE, false, false); // Setting [correcting] bias offset

		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = eq1conn[i]->Weight();
			w2 = eq2conn[i]->Weight();
			if (proj_spread(w1, w2) > -1e-3f) { // Same signs, suppose single feild usage
				m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f, false, false); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
			} else { // Different signs, suppose possible both feilds usage
				// Turning into positive to direct to illinear part (instead of just linear)
				m1proj.emplace_back((std::fabs(w1) + std::fabs(w2)) / 2.0f, w1 > 0, w2 > 0); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
			}
		}

		float rotsign = ((static_cast<int8_t>(rotflag) << 1) - 1);
		float diff;
		// Bias handling
		{
			w1 = std::get<0>(m1proj[0]);
			w2 = m1conn[0]->Weight();
			diff = std::fabs(w2 - w1); // New - Old
			if (proj_spread(w1, w2) > -1e-2f) { // Suppose bias meaning does not changed
				eq1conn[0]->Weight(eq1conn[0]->Weight() + rotsign * diff);
				eq2conn[0]->Weight(eq2conn[0]->Weight() - rotsign * diff); // eq2conn[0]->Weight() + -rotsign * diff
			} else { // Bias became to be something strange... apparently, the network has decided to use just second feild... unwanted result.
				// Regenerating biases! Serious intervention, possible convergence problems.
				eq1conn[0]->Weight(rotsign * (w2 + M1_MAX_VALUE));
				eq2conn[0]->Weight(rotsign * (M1_MAX_VALUE - w2)); // eq2conn[0]->Weight(-rotsign * (w2 - M1_MAX_VALUE))
			}
		}
		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = std::get<0>(m1proj[i]);
			w2 = m1conn[i]->Weight();
			diff = std::fabs(w2 - w1); // New - Old
			if (proj_spread(w1, w2) > -1e-2f) { // Suppose bias meaning does not changed
				if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single feild user
					eq1conn[i]->Weight(eq1conn[i]->Weight() + rotsign * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() - rotsign * diff); // eq2conn[i]->Weight() + -rotsign * diff
				} else { // Both feilds user
					eq1conn[i]->Weight(eq1conn[i]->Weight() + ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() + ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
				}
			} else { // Weight meaning [maybe] changed
				if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single feild user
					// Swapping the feilds
					eq1conn[i]->Weight(eq1conn[i]->Weight() - rotsign * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() + rotsign * diff);
				} else { // Both feilds user
					// Swapping the feilds
					eq1conn[i]->Weight(eq1conn[i]->Weight() - ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() - ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
				}
			}
		}
	};

	nn::utils::Monotonic2FeildsProjectingAccessory stand_m1_proj(&m1_layer1, &stand_layer1, proj2to1, proj1to2);

	nn::BasicOutsErrorSetter::ErrorCalcMSE mse_calculator;
	nn::BasicOutsErrorSetter nerr_setter(&mse_calculator, perfect_out.size());

	nn::FwdBackPropGuider stand_learnguider({ &stand_layer1, &stand_layer2 }, &nerr_setter);
	nn::FwdBackPropGuider m1_learnguider({ &m1_layer1 }, &nerr_setter);

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