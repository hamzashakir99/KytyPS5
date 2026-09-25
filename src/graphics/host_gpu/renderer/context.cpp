#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

// Device-lost diagnostics (VK_NV_device_diagnostic_checkpoints). Every debug-info update drops a
// checkpoint whose marker points at a copy of that info, so after a lost device the driver
// reports the last work the GPU actually reached.
namespace {

struct CheckpointRecord {
	uint64_t sequence;
	uint64_t submit_id;
	uint64_t arg4;
	uint64_t cs_addr;
	uint64_t ps_addr;
	uint64_t gs_addr;
	uint32_t op;
	uint32_t args[4];
};

constexpr size_t CHECKPOINT_RECORD_COUNT = size_t {1} << 16;

CheckpointRecord      g_checkpoint_records[CHECKPOINT_RECORD_COUNT];
std::atomic<uint64_t> g_checkpoint_sequence {0};
vk::Queue             g_checkpoint_queue = nullptr;

} // namespace

void EnableDeviceCheckpoints(vk::Queue queue) {
	g_checkpoint_queue = queue;
}

void PrintDeviceCheckpoints() {
	if (g_checkpoint_queue == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		std::printf("device checkpoints: unavailable\n");
		return;
	}
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(g_checkpoint_queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> checkpoints(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(g_checkpoint_queue, &count,
	                                                         checkpoints.data());
	std::printf("device checkpoints (%u, last work each pipeline stage reached):\n", count);
	for (uint32_t i = 0; i < count; i++) {
		const auto* record = static_cast<const CheckpointRecord*>(checkpoints[i].pCheckpointMarker);
		if (record == nullptr) {
			continue;
		}
		std::printf("  stage=0x%08x seq=%" PRIu64 " op=%u submit=%" PRIu64
		            " args=%u,%u,%u,%u,0x%016" PRIx64 " cs=%016" PRIx64 " ps=%016" PRIx64
		            " gs=%016" PRIx64 "\n",
		            static_cast<uint32_t>(checkpoints[i].stage), record->sequence, record->op,
		            record->submit_id, record->args[0], record->args[1], record->args[2],
		            record->args[3], record->arg4, record->cs_addr, record->ps_addr,
		            record->gs_addr);
		// Save the compute shader that follows the checkpoint so it can be matched to its dump.
		if (record->op == static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect) ||
		    record->op == static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect)) {
			constexpr uint32_t    S_ENDPGM  = 0xbf810000u;
			constexpr size_t      MAX_WORDS = 64 * 1024;
			std::vector<uint32_t> code;
			uint32_t              word = 0;
			while (code.size() < MAX_WORDS &&
			       LibKernel::Memory::TryReadGpuCleanBacking(record->cs_addr + code.size() * 4,
			                                                 &word, sizeof(word))) {
				code.push_back(word);
				if (word == S_ENDPGM) {
					break;
				}
			}
			char name[64];
			std::snprintf(name, sizeof(name), "device_lost_cs_%016" PRIx64 ".bin",
			              record->cs_addr);
			if (auto* file = std::fopen(name, "wb"); file != nullptr) {
				std::fwrite(code.data(), sizeof(uint32_t), code.size(), file);
				std::fclose(file);
				std::printf("  saved %zu words of the compute shader to %s\n", code.size(), name);
			}
		}
	}
	std::fflush(stdout);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;

	if (g_checkpoint_queue != nullptr && m_buffer != nullptr) {
		const uint64_t sequence = g_checkpoint_sequence.fetch_add(1, std::memory_order_relaxed);
		auto&          record   = g_checkpoint_records[sequence % CHECKPOINT_RECORD_COUNT];
		record                  = {sequence, submit_id, arg4, 0, 0, 0, op, {arg0, arg1, arg2, arg3}};
		if (m_shaders != nullptr) {
			record.cs_addr = m_shaders->GetCs().cs_regs.data_addr;
			record.ps_addr = m_shaders->GetPs().ps_regs.data_addr;
			record.gs_addr = m_shaders->GetVs().gs_regs.data_addr;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(m_buffer, &record);
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
