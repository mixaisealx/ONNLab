#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"

#include <functional>

namespace nn::utils
{
	class Monotonic2FeildsProjectingAccessory {
	public:
		using ProjectorTwoToOne = void(interfaces::NBI *const non_monotonic, const interfaces::NBI *const equvivalent_1, const interfaces::NBI *const equvivalent_2);
		using ProjectorOneToTwo = void(const interfaces::NBI *const non_monotonic, interfaces::NBI *const equvivalent_1, interfaces::NBI *const equvivalent_2);

		enum LayerAlignment { MOD2, HALF };

		inline Monotonic2FeildsProjectingAccessory(interfaces::BasicLayerInterface *non_monothonic_layer,
											interfaces::BasicLayerInterface *monothonic_equvivalent_layer,
											std::function<ProjectorTwoToOne> projector2to1,
											std::function<ProjectorOneToTwo> projector1to2,
											LayerAlignment alignment = LayerAlignment::HALF):projector2to1(projector2to1), projector1to2(projector1to2) {
			unsigned mnls = non_monothonic_layer->Neurons().size();

			if (monothonic_equvivalent_layer->Neurons().size() != mnls * 2)
				throw std::logic_error("Count of neurons in eqvivalent layer must be 2 times bigger than in non-motonic!");

			auto &vec_nmn = non_monothonic_layer->Neurons();
			auto &vec_men = monothonic_equvivalent_layer->Neurons();

			if (alignment == LayerAlignment::HALF) {
				for (unsigned i = 0; i != mnls; ++i) {
					eqviv.emplace_back(Equvivalence{ vec_nmn[i] , vec_men[i], vec_men[mnls + i] });
				}
			} else {
				for (unsigned i = 0; i != mnls; ++i) {
					eqviv.emplace_back(Equvivalence{ vec_nmn[i] , vec_men[i << 1], vec_men[(i << 1) + 1] });
				}
			}
		}

		inline void Perform1to2DiffTransfer() {
			for (auto &elem : eqviv) {
				projector1to2(elem.non_monotonic, elem.equvivalent_part_1, elem.equvivalent_part_2);
			}
		}

		inline void Perform2to1LossyCompression() {
			for (auto &elem : eqviv) {
				projector2to1(elem.non_monotonic, elem.equvivalent_part_1, elem.equvivalent_part_2);
			}
		}
	private:
		struct Equvivalence {
			interfaces::NBI *non_monotonic;
			interfaces::NBI *equvivalent_part_1, *equvivalent_part_2;
		};

		std::function<ProjectorOneToTwo> projector1to2;
		std::function<ProjectorTwoToOne> projector2to1;

		std::vector<Equvivalence> eqviv;
	};
}
