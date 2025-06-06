#pragma once
#include "NNBasicsInterfaces.h"
#include "Monotonic2FieldsHeuristicsI.h"
#include "LimitedNeuronI.h"

#include <cmath>

namespace nn
{
	class Monotonic2FieldsHeuristicsEqExV2 : public nn::interfaces::Monotonic2FieldsHeuristicsI {
		const float spread_limit_2to1;
		const float spread_limit_1to2;

		static inline float projector_spread(float w1, float w2) {
			static const float EPS = 1e-8f; // For numeric stability
			// (min(|w1|,|w2|)/(max(|w1|,|w2|)+EPS))^2 * sign(w1*w2)
			int8_t sign = (static_cast<int8_t>(w1 > 0.0f == w2 > 0.0f) << 1) - 1;
			w1 = std::fabs(w1);
			w2 = std::fabs(w2);
			w1 = w1 > w2 ? w2 / (w1 + EPS) : w1 / (w2 + EPS);
			return w1 * w1 * static_cast<float>(sign);
		};
	public:
		//**
		// Equivalent excnahge heuristic V2
		// NM1_MAX_VALUE - max value of nm1ReLU on current layer
		// both_side_cutoff - value from -1.0 to 1.0, both side detector floor level, lower value -> more sensetive (results in more both side users)
		// meaning_change_level - value from -1.0 to 1.0, weigth meaning change detector lever, lower value -> less sensetive (results in more loose detection of weight changing)
		// */
		Monotonic2FieldsHeuristicsEqExV2(float both_side_cutoff = -0.5f, float meaning_change_level = -5e-3f):spread_limit_2to1(both_side_cutoff), spread_limit_1to2(meaning_change_level){}

		void Perform2to1(nn::interfaces::NBI *const m1, const nn::interfaces::NBI *const eq1, const nn::interfaces::NBI *const eq2, size_t) override {
			auto &eq1conn = const_cast<nn::interfaces::NBI *>(eq1)->InputConnections();
			auto &eq2conn = const_cast<nn::interfaces::NBI *>(eq2)->InputConnections();
			auto &m1conn = m1->InputConnections();
			float NM1_MAX_VALUE = dynamic_cast<nn::interfaces::LimitedNeuronI *>(m1)->UpperLimitValue();

			float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
			bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better

			m1conn[0]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f - NM1_MAX_VALUE); // Setting [correcting] bias offset
			for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
				w1 = eq1conn[i]->Weight();
				w2 = eq2conn[i]->Weight();
				if (projector_spread(w1, w2) > spread_limit_2to1) { // Same signs OR very different modules, suppose single field usage
					m1conn[i]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
				} else { // Different signs AND enought same modules, suppose possible both fields usage
					// Turning into positive to direct to nonmonotonic part (instead of just monotonic)
					m1conn[i]->Weight((std::fabs(w1) + std::fabs(w2)) / 2.0f); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
				}
			}
		}
		void Perform1to2(const nn::interfaces::NBI *const m1, nn::interfaces::NBI *const eq1, nn::interfaces::NBI *const eq2, size_t) override {
			auto &eq1conn = eq1->InputConnections();
			auto &eq2conn = eq2->InputConnections();
			auto &m1conn = const_cast<nn::interfaces::NBI *>(m1)->InputConnections();
			float NM1_MAX_VALUE = dynamic_cast<const nn::interfaces::LimitedNeuronI *>(m1)->UpperLimitValue();

			std::vector<std::tuple<float, bool, bool>> m1proj;
			m1proj.reserve(m1conn.size());

			float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
			bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better
			m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f - NM1_MAX_VALUE, false, false); // Setting [correcting] bias offset

			for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
				w1 = eq1conn[i]->Weight();
				w2 = eq2conn[i]->Weight();
				if (projector_spread(w1, w2) > spread_limit_2to1) { // Same signs OR very different modules, suppose single field usage
					m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f, false, false); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
				} else { // Different signs AND enought same modules, suppose possible both fields usage
					// Turning into positive to direct to nonmonotonic part (instead of just monotonic)
					m1proj.emplace_back((std::fabs(w1) + std::fabs(w2)) / 2.0f, w1 > 0, w2 > 0); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
				}
			}

			float rotsign = static_cast<int8_t>((static_cast<int8_t>(rotflag) << 1) - 1);
			float diff;
			// Bias handling
			{
				w1 = std::get<0>(m1proj[0]);
				w2 = m1conn[0]->Weight();
				diff = std::fabs(w2 - w1); // New - Old
				if (projector_spread(w1, w2) > spread_limit_1to2) { // Suppose bias meaning does not changed
					eq1conn[0]->Weight(eq1conn[0]->Weight() + rotsign * diff);
					eq2conn[0]->Weight(eq2conn[0]->Weight() - rotsign * diff); // eq2conn[0]->Weight() + -rotsign * diff
				} else { // Bias became to be something strange... apparently, the network has decided to use just second field... unwanted result.
					// Regenerating biases! Serious intervention, possible convergence problems.
					eq1conn[0]->Weight(rotsign * (w2 + NM1_MAX_VALUE));
					eq2conn[0]->Weight(rotsign * (NM1_MAX_VALUE - w2)); // eq2conn[0]->Weight(-rotsign * (w2 - NM1_MAX_VALUE))
				}
			}
			for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
				w1 = std::get<0>(m1proj[i]);
				w2 = m1conn[i]->Weight();
				diff = std::fabs(w2 - w1); // New - Old
				if (projector_spread(w1, w2) > spread_limit_1to2) { // Suppose bias meaning does not changed
					if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single field user
						eq1conn[i]->Weight(eq1conn[i]->Weight() + rotsign * diff);
						eq2conn[i]->Weight(eq2conn[i]->Weight() - rotsign * diff); // eq2conn[i]->Weight() + -rotsign * diff
					} else { // Both fields user
						eq1conn[i]->Weight(eq1conn[i]->Weight() + ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
						eq2conn[i]->Weight(eq2conn[i]->Weight() + ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
					}
				} else { // Weight meaning [maybe] changed
					if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single field user
						// Swapping the fields
						eq1conn[i]->Weight(eq1conn[i]->Weight() - rotsign * diff);
						eq2conn[i]->Weight(eq2conn[i]->Weight() + rotsign * diff);
					} else { // Both fields user
						// Swapping the fields
						eq1conn[i]->Weight(eq1conn[i]->Weight() - ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
						eq2conn[i]->Weight(eq2conn[i]->Weight() - ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
					}
				}
			}
		}
	};
}
