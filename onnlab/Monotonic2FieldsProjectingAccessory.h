#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"
#include "Monotonic2FieldsHeuristicsI.h"

#include <functional>

namespace nn
{
	class Monotonic2FieldsProjectingAccessory {
		class Monotonic2FieldsHeuristicsWrapper : public nn::interfaces::Monotonic2FieldsHeuristicsI {
			std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorTwoToOne> projector2to1;
			std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorOneToTwo> projector1to2;
		public:
			Monotonic2FieldsHeuristicsWrapper(std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorTwoToOne> projector2to1,
											  std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorOneToTwo> projector1to2):
				projector2to1(projector2to1), projector1to2(projector1to2) {}
		
			void Perform2to1(interfaces::NBI *const non_monotonic, const interfaces::NBI *const equvivalent_1, const interfaces::NBI *const equvivalent_2, size_t) override {
				projector2to1(non_monotonic, equvivalent_1, equvivalent_2);
			}
			void Perform1to2(const interfaces::NBI *const non_monotonic, interfaces::NBI *const equvivalent_1, interfaces::NBI *const equvivalent_2, size_t) override {
				projector1to2(non_monotonic, equvivalent_1, equvivalent_2);
			}
		};

		Monotonic2FieldsProjectingAccessory(const Monotonic2FieldsProjectingAccessory &) = delete;
		Monotonic2FieldsProjectingAccessory &operator=(const Monotonic2FieldsProjectingAccessory &) = delete;
	public:
		enum LayerAlignment { MOD2, HALF };

		inline Monotonic2FieldsProjectingAccessory(interfaces::BasicLayerInterface *non_monothonic_layer,
												   interfaces::BasicLayerInterface *monothonic_equvivalent_layer,
												   nn::interfaces::Monotonic2FieldsHeuristicsI *projector,
												   LayerAlignment alignment = LayerAlignment::HALF):projector(projector), wrapper(nullptr) {
			Init(non_monothonic_layer, monothonic_equvivalent_layer, alignment);
		}

		inline Monotonic2FieldsProjectingAccessory(interfaces::BasicLayerInterface *non_monothonic_layer,
											interfaces::BasicLayerInterface *monothonic_equvivalent_layer,
											std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorTwoToOne> projector2to1,
											std::function<nn::interfaces::Monotonic2FieldsHeuristicsI::ProjectorOneToTwo> projector1to2,
											LayerAlignment alignment = LayerAlignment::HALF) {
			projector = wrapper = new Monotonic2FieldsHeuristicsWrapper(projector2to1, projector1to2);
			Init(non_monothonic_layer, monothonic_equvivalent_layer, alignment);
		}

		inline void Perform1to2DiffTransfer() {
			for (size_t i = 0,lim = equiv.size(); i != lim; ++i) {
				auto &elem = equiv[i];
				projector->Perform1to2(elem.non_monotonic, elem.equvivalent_part_1, elem.equvivalent_part_2, i);
			}
		}

		inline void Perform2to1LossyCompression() {
			for (size_t i = 0, lim = equiv.size(); i != lim; ++i) {
				auto &elem = equiv[i];
				projector->Perform2to1(elem.non_monotonic, elem.equvivalent_part_1, elem.equvivalent_part_2, i);
			}
		}

		inline virtual ~Monotonic2FieldsProjectingAccessory() {
			delete wrapper;
		}
	private:
		inline void Init(interfaces::BasicLayerInterface *non_monothonic_layer,
						 interfaces::BasicLayerInterface *monothonic_equvivalent_layer,
						 LayerAlignment alignment) {
			unsigned mnls = non_monothonic_layer->Neurons().size();

			if (monothonic_equvivalent_layer->Neurons().size() != mnls * 2)
				throw std::exception("Count of neurons in equivalent layer must be 2 times bigger than in non-motonic!");

			auto &vec_nmn = non_monothonic_layer->Neurons();
			auto &vec_men = monothonic_equvivalent_layer->Neurons();

			if (alignment == LayerAlignment::HALF) {
				for (unsigned i = 0; i != mnls; ++i) {
					equiv.emplace_back(Equvivalence{ vec_nmn[i] , vec_men[i], vec_men[mnls + i] });
				}
			} else {
				for (unsigned i = 0; i != mnls; ++i) {
					equiv.emplace_back(Equvivalence{ vec_nmn[i] , vec_men[i << 1], vec_men[(i << 1) + 1] });
				}
			}
		}

		struct Equvivalence {
			interfaces::NBI *non_monotonic;
			interfaces::NBI *equvivalent_part_1, *equvivalent_part_2;
		};

		Monotonic2FieldsHeuristicsWrapper *wrapper;
		nn::interfaces::Monotonic2FieldsHeuristicsI *projector;

		std::vector<Equvivalence> equiv;
	};
}
