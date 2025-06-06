#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"
#include "BasicWghOptI.h"

#include <cstdlib>
#include <concepts>
#include <vector>
#include <span>
#include <functional>

namespace nn
{
	template<interfaces::ConnectionInherit ConnectionT>
	class SparceLayerStaticConnectomHolder2Mult : public interfaces::BasicWeightOptimizableInterface {
		ConnectionT *storage_base, *storage_end;
		std::span<ConnectionT> obj_view;

		SparceLayerStaticConnectomHolder2Mult(const SparceLayerStaticConnectomHolder2Mult &) = delete;
		SparceLayerStaticConnectomHolder2Mult &operator=(const SparceLayerStaticConnectomHolder2Mult &) = delete;
	public:
		using ConnectionEmplacer = void(ConnectionT*const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to);

		enum LayerAlignment { MOD2, HALF };

		SparceLayerStaticConnectomHolder2Mult(interfaces::BasicLayerInterface *layer1,
								  interfaces::BasicLayerInterface *layer2,
								  std::function<ConnectionEmplacer> emplacer,
								  LayerAlignment alignment = LayerAlignment::HALF) {
			
			if (layer1->Neurons().size() != layer2->Neurons().size() * 2)
				throw std::exception("Count of neurons in layer1 must be 2 times bigger than in layer2!");

			size_t connections_count = layer1->Neurons().size();
			storage_base = reinterpret_cast<ConnectionT*>(malloc(connections_count * sizeof(ConnectionT))); // Yes, I know about possible nullptr, BUT it is only lab code
			storage_end = storage_base + connections_count;

			ConnectionT *mem_ptr = storage_base;

			if (alignment == LayerAlignment::HALF) {
				auto &from = layer1->Neurons();
				auto &to = layer2->Neurons();
				unsigned outsz = layer2->Neurons().size();
				for (unsigned i = 0; i != outsz; ++i) {
					emplacer(mem_ptr, from[i], to[i]);
					++mem_ptr;
					emplacer(mem_ptr, from[outsz + i], to[i]);
					++mem_ptr;
				}
			} else {
				auto iter = layer1->Neurons().cbegin();
				for (auto to : layer2->Neurons()) {
					for (uint8_t i = 2; i; --i) {
						emplacer(mem_ptr, *iter, to);
						++iter;
						++mem_ptr;
					}
				}
			}
			
			obj_view = std::span<ConnectionT>{ storage_base, connections_count };
		}

		const std::span<ConnectionT> &ConnectionsInside() {
			return obj_view;
		}

		void WeightOptimReset() override {
			interfaces::BasicWeightOptimizableInterface *wopt;
			for (ConnectionT *curr = storage_base; curr != storage_end; ++curr) {
				wopt = dynamic_cast<nn::interfaces::BasicWeightOptimizableInterface *>(curr);
				if (wopt) {
					wopt->WeightOptimReset();
				}
			}
		}

		void WeightOptimDoUpdate(float gradient) override {
			interfaces::BasicWeightOptimizableInterface *wopt;
			for (ConnectionT *curr = storage_base; curr != storage_end; ++curr) {
				wopt = dynamic_cast<nn::interfaces::BasicWeightOptimizableInterface *>(curr);
				if (wopt) {
					wopt->WeightOptimDoUpdate(gradient);
				}
			}
		}

		~SparceLayerStaticConnectomHolder2Mult() {
			for (ConnectionT *curr = storage_base; curr != storage_end; ++curr) {
				dynamic_cast<nn::interfaces::CBI *>(curr)->~ConnectionBasicInterface();
			}
			free(storage_base);
		}

		struct CBIterator {
			using iterator_category = std::bidirectional_iterator_tag;
			using value_type = interfaces::ConnectionBasicInterface *;
			using pointer = interfaces::ConnectionBasicInterface *; // To don't lose casted to CBI* pointer
			using reference = interfaces::ConnectionBasicInterface *; // To don't lose casted to CBI* pointer
			CBIterator(ConnectionT *ptr): ptr(ptr) {}
			ConnectionT *raw() {
				return ptr;
			}
			interfaces::ConnectionBasicInterface *operator*() {
				return dynamic_cast<interfaces::CBI *>(ptr);
			}
			CBIterator &operator++() {
				++ptr;
				return *this;
			}
			CBIterator operator++(int) {
				auto tmp = *this;
				++*this;
				return tmp;
			}
			CBIterator &operator--() {
				--ptr;
				return *this;
			}
			CBIterator operator--(int) {
				auto tmp = *this;
				--*this;
				return tmp;
			}
			bool operator==(const CBIterator &other) const {
				return ptr == other.ptr;
			}
			bool operator!=(const CBIterator &other) const {
				return ptr != other.ptr;
			}
		private:
			ConnectionT *ptr;
		};

		CBIterator begin() {
			return CBIterator(storage_base);
		}

		CBIterator end() {
			return CBIterator(storage_end);
		}
	};
}