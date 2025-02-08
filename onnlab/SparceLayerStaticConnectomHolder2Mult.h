#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"

#include <cstdlib>
#include <concepts>
#include <vector>
#include <span>
#include <functional>

namespace nn
{
	template<interfaces::ConnectionInherit ConnectionT>
	class SparceLayerStaticConnectomHolder2Mult {
		ConnectionT *storage_base, *storage_end;
		std::span<ConnectionT> obj_view;
	public:
		using ConnectionEmplacer = void(ConnectionT*const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to);

		SparceLayerStaticConnectomHolder2Mult(interfaces::BasicLayerInterface *layer1,
								  interfaces::BasicLayerInterface *layer2,
								  std::function<ConnectionEmplacer> emplacer) {
			
			if (layer1->Neurons().size() != layer2->Neurons().size() * 2)
				throw std::logic_error("Count of neurons in layer1 must be 2 times bigger than in layer2!");

			size_t connections_count = layer1->Neurons().size();
			storage_base = reinterpret_cast<ConnectionT*>(malloc(connections_count * sizeof(ConnectionT))); // Yes, I know about possible nullptr, BUT it is only lab code
			storage_end = storage_base + connections_count;

			ConnectionT *mem_ptr = storage_base;

			auto iter = layer1->Neurons().cbegin();
			for (auto to : layer2->Neurons()) {
				for (uint8_t i = 2; i; --i) {
					emplacer(mem_ptr, *iter, to);
					++iter;
					++mem_ptr;
				}
			}
			
			obj_view = std::span<ConnectionT>{ storage_base, connections_count };
		}

		const std::span<ConnectionT> &ConnectionsInside() {
			return obj_view;
		}

		~SparceLayerStaticConnectomHolder2Mult() {
			for (ConnectionT *curr = storage_base; curr != storage_end; ++curr) {
				dynamic_cast<nn::interfaces::CBI *>(curr)->~ConnectionBasicInterface();
			}
			free(storage_base);
		}
	};
}