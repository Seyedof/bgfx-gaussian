/*
 * Copyright 2023 Ali Seyedof. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#include <bx/allocator.h>
#include <bx/debug.h>
#include <bx/math.h>
#include <bx/thread.h>
#include <thread>
#include <future>
#include <array>
#include <vector>
#include <atomic>
#include "common.h"
#include "bgfx_utils.h"
#include "imgui/imgui.h"
#include "camera.h"

namespace
{

struct PosVertex
{
	float m_x;
	float m_y;

	static void init()
	{
		ms_layout
			.begin()
			.add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
			.end();
	}

	static bgfx::VertexLayout ms_layout;
};

bgfx::VertexLayout PosVertex::ms_layout;

struct InstanceDataVertex
{
	float m_r;
	float m_g;
	float m_b;
	float m_a;
	float m_cx;
	float m_cy;
	float m_cz;
	float m_cw;
	float m_cova_x;
	float m_cova_y;
	float m_cova_z;
	float m_cova_w;
	float m_covb_x;
	float m_covb_y;
	float m_covb_z;
	float m_covb_w;

	static void init()
	{
		ms_layoutInstance
			.begin()
			.add(bgfx::Attrib::TexCoord0, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord1, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord2, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord3, 4, bgfx::AttribType::Float)
			.end();
	}

	static bgfx::VertexLayout ms_layoutInstance;
};

bgfx::VertexLayout InstanceDataVertex::ms_layoutInstance;

struct SplatFileRecord {
	float center[3];
	float scale[3];
	uint32_t color;
	uint8_t i;
	uint8_t j;
	uint8_t k;
	uint8_t l;
};

struct GaussianSplatData {
	uint32_t			m_vertexCount = 0;
	InstanceDataVertex* m_buffer = nullptr;
	bgfx::DynamicVertexBufferHandle m_vbh = BGFX_INVALID_HANDLE;
	bgfx::InstanceDataBuffer m_idb;
};

class ExampleGaussianSplatting : public entry::AppI
{
public:
	ExampleGaussianSplatting(const char* _name, const char* _description, const char* _url)
		: entry::AppI(_name, _description, _url)
	{
	}

	void init(int32_t _argc, const char* const* _argv, uint32_t _width, uint32_t _height) override
	{
		Args args(_argc, _argv);

		m_width  = _width;
		m_height = _height;
		m_debug  = BGFX_DEBUG_NONE;
		m_reset = 0;

		bgfx::Init init;
		//init.type = bgfx::RendererType::Direct3D11;
		init.type = bgfx::RendererType::OpenGL;
		//init.type = bgfx::RendererType::Vulkan;
		//init.type     = args.m_type;
		init.vendorId = args.m_pciId;
		init.platformData.nwh  = entry::getNativeWindowHandle(entry::kDefaultWindowHandle);
		init.platformData.ndt  = entry::getNativeDisplayHandle();
		init.platformData.type = entry::getNativeWindowHandleType();
		init.resolution.width  = m_width;
		init.resolution.height = m_height;
		init.resolution.reset  = m_reset;
		bgfx::init(init);

		// Enable m_debug text.
		bgfx::setDebug(m_debug);

		// Set view 0 clear state.
		bgfx::setViewClear(0
			, BGFX_CLEAR_COLOR
			, 0x00000000
			, 1.0f
			, 0
			);

		// Create vertex stream declaration.
		PosVertex::init();
		InstanceDataVertex::init();

		// Create program from shaders.
		m_gaussianProgram = loadProgram("vs_3dgaussian", "fs_3dgaussian");

		// Imgui.
		imguiCreate();

		m_timeOffset = bx::getHPCounter();

		FILE* file = fopen("binary/train.splat", "rb");
		//FILE* file = fopen("binary/truck.splat", "rb");
		fseek(file, 0, SEEK_END);
		size_t fileSize = ftell(file);
		size_t splatCount = fileSize / sizeof(SplatFileRecord);
		std::vector<SplatFileRecord> rawFileData(splatCount);
		fseek(file, 0, SEEK_SET);
		fread(rawFileData.data(), sizeof(SplatFileRecord), splatCount, file);
		fclose(file);

		m_splatFileData.resize(splatCount);
		for (size_t i = 0; i < splatCount; ++i) {
			for (int j = 0; j < 3; ++j) {
				if (rawFileData[i].center[j] < bboxMin[j])
					bboxMin[j] = rawFileData[i].center[j];
				if (rawFileData[i].center[j] > bboxMax[j])
					bboxMax[j] = rawFileData[i].center[j];
			}
			memcpy(&m_splatFileData[i].m_cx, rawFileData[i].center, 3 * sizeof(float));
			m_splatFileData[i].m_r = ((rawFileData[i].color >> 0) & 0xff)  / 255.0f;
			m_splatFileData[i].m_g = ((rawFileData[i].color >> 8) & 0xff)  / 255.0f;
			m_splatFileData[i].m_b = ((rawFileData[i].color >> 16) & 0xff) / 255.0f;
			m_splatFileData[i].m_a = ((rawFileData[i].color >> 24) & 0xff) / 255.0f;

			float scale[3];
			memcpy(scale,  rawFileData[i].scale, 3 * sizeof(float));

			float quat[4];
			quat[0] = ((float)rawFileData[i].i - 128.0f) / 128.0f;
			quat[1] = ((float)rawFileData[i].j - 128.0f) / 128.0f;
			quat[2] = ((float)rawFileData[i].k - 128.0f) / 128.0f;
			quat[3] = ((float)rawFileData[i].l - 128.0f) / 128.0f;

			float rot[4];
			memcpy(rot, quat, 4 * sizeof(float));

			const float matRot[9] = {
				1.0f - 2.0f * (rot[2] * rot[2] + rot[3] * rot[3]),
				2.0f * (rot[1] * rot[2] + rot[0] * rot[3]),
				2.0f * (rot[1] * rot[3] - rot[0] * rot[2]),

				2.0f * (rot[1] * rot[2] - rot[0] * rot[3]),
				1.0f - 2.0f * (rot[1] * rot[1] + rot[3] * rot[3]),
				2.0f * (rot[2] * rot[3] + rot[0] * rot[1]),

				2.0f * (rot[1] * rot[3] + rot[0] * rot[2]),
				2.0f * (rot[2] * rot[3] - rot[0] * rot[1]),
				1.0f - 2.0f * (rot[1] * rot[1] + rot[2] * rot[2]),
			};

			// Compute the matrix product of S and R (M = S * R)
			const float matSR[9] = {
				scale[0] * matRot[0],
				scale[0] * matRot[1],
				scale[0] * matRot[2],
				scale[1] * matRot[3],
				scale[1] * matRot[4],
				scale[1] * matRot[5],
				scale[2] * matRot[6],
				scale[2] * matRot[7],
				scale[2] * matRot[8],
			};

			m_splatFileData[i].m_cova_x = matSR[0] * matSR[0] + matSR[3] * matSR[3] + matSR[6] * matSR[6];
			m_splatFileData[i].m_cova_y = matSR[0] * matSR[1] + matSR[3] * matSR[4] + matSR[6] * matSR[7];
			m_splatFileData[i].m_cova_z = matSR[0] * matSR[2] + matSR[3] * matSR[5] + matSR[6] * matSR[8];

			m_splatFileData[i].m_covb_x = matSR[1] * matSR[1] + matSR[4] * matSR[4] + matSR[7] * matSR[7];
			m_splatFileData[i].m_covb_y = matSR[1] * matSR[2] + matSR[4] * matSR[5] + matSR[7] * matSR[8];
			m_splatFileData[i].m_covb_z = matSR[2] * matSR[2] + matSR[5] * matSR[5] + matSR[8] * matSR[8];
		}
		rawFileData.clear();

		bx::Vec3 v2(bboxMax[0] - bboxMin[0], bboxMax[1] - bboxMin[1], bboxMax[2] - bboxMin[2]);
		m_range = bx::length(v2);

		float fov = 60.0;
		float aspect = (float)m_width / (float)m_height;
		float fx = (float)m_width / 2.0f / tanf(bx::toRad(fov / 2.0f));
		float fy = (float)m_height * aspect / 2.0f / tanf(bx::toRad(fov / 2.0f));

		m_focal[0] = fx;
		m_focal[1] = fy;

		bx::mtxProj(m_projMtx, 45.0f, float(m_width) / float(m_height), 0.2f, 200.0f, false);

		cameraCreate();
		cameraSetPosition(bx::Vec3(-3, 0, -4));

		for (int i = 0; i < 2; i++) {
			m_splatData[i].m_vertexCount = static_cast<uint32_t>(splatCount);
			m_splatData[i].m_buffer = new InstanceDataVertex[splatCount];
			memset(m_splatData[i].m_buffer, 0, sizeof(InstanceDataVertex) * splatCount);
			m_splatData[i].m_vbh = bgfx::createDynamicVertexBuffer(bgfx::makeRef(m_splatData[i].m_buffer, (uint32_t)splatCount * sizeof(InstanceDataVertex)), InstanceDataVertex::ms_layoutInstance);
		}

		const static PosVertex triangleVertices[4] = { 2, -2, -2, -2, 2, 2, -2, 2 };
		m_vbh = bgfx::createVertexBuffer(bgfx::makeRef(triangleVertices, sizeof(PosVertex) * 4), PosVertex::ms_layout);

		u_focal = bgfx::createUniform("u_focal", bgfx::UniformType::Vec4);

		m_oldWidth  = 0;
		m_oldHeight = 0;
		m_oldReset  = m_reset;
	}

	virtual int shutdown() override
	{
		// Cleanup.
		cameraDestroy();
		imguiDestroy();

		for (int i = 0; i < 2; i++) {
			m_splatData[i].m_vertexCount = 0;
			if (m_splatData[i].m_buffer) {
				delete m_splatData[i].m_buffer;
			}
			if (bgfx::isValid(m_splatData[i].m_vbh))
			{
				bgfx::destroy(m_splatData[i].m_vbh);
			} 
		}

		if (bgfx::isValid(m_vbh) )
		{
			bgfx::destroy(m_vbh);
		}

		/// When data is passed to bgfx via makeRef we need to make
		/// sure library is done with it before freeing memory blocks.
		bgfx::frame();

		//bx::AllocatorI* allocator = entry::getAllocator();
		//bx::free(allocator, m_terrain.m_vertices);
		//bx::free(allocator, m_terrain.m_indices);
		//bx::free(allocator, m_terrain.m_heightMap);

		// Shutdown bgfx.
		bgfx::shutdown();

		return 0;
	}

	void SortGaussians(float* view)
	{
		if (!m_splatData[m_curBuffer].m_vertexCount) {
			return;
		}

		m_isSorting = true;

		int nextBuffer = 1 - m_curBuffer;
		uint32_t vertexCount = m_splatData[nextBuffer].m_vertexCount;

		std::vector<uint32_t> depthIndex(vertexCount+1);
		InstanceDataVertex* vertexBuffer = (InstanceDataVertex*)m_splatData[nextBuffer].m_buffer;

		float maxDepth = -FLT_MAX;
		float minDepth = FLT_MAX;
		static std::vector<int32_t> sizeList(vertexCount);
		bx::Vec3 zAxis(view[2] * 4096.0f, view[6] * 4096.0f, view[10] * 4096.0f);
		for (uint32_t i = 0; i < vertexCount; ++i) {
			float depth = bx::dot(bx::Vec3(m_splatFileData[i].m_cx, m_splatFileData[i].m_cy, m_splatFileData[i].m_cz), zAxis);
			sizeList[i] = static_cast<uint32_t>(depth);
			if (depth > maxDepth) maxDepth = depth;
			if (depth < minDepth) minDepth = depth;
		}

		float depthInv = (256 * 256) / (maxDepth - minDepth);
		static std::array<uint32_t, 256 * 256 + 1> counts0;
		memset(counts0.data(), 0, (256 * 256 + 1) * sizeof(uint32_t));
		for (uint32_t i = 0; i < vertexCount; ++i) {
			sizeList[i] = static_cast<uint32_t>(((sizeList[i] - minDepth) * depthInv));
			counts0[sizeList[i]]++;
		}

		static std::array<uint32_t, 256 * 256 + 1> starts0;
		starts0[0] = 0;
		for (uint32_t i = 1; i < 256 * 256; ++i) {
			starts0[i] = starts0[i - 1] + counts0[i - 1];
		}
		for (uint32_t i = 0; i < vertexCount; ++i) {
			depthIndex[starts0[sizeList[i]]++] = i;
		}

		for (uint32_t j = 0; j < vertexCount; ++j) {
			const uint32_t i = depthIndex[j];

			vertexBuffer[j] = m_splatFileData[i];
		}

		bgfx::ReleaseFn releaseFn = [](void* data, void* bufferPtr) {
			BX_UNUSED(data);
			BX_UNUSED(bufferPtr);
			ExampleGaussianSplatting* example = (ExampleGaussianSplatting*)bufferPtr;
			example->m_curBuffer = 1 - example->m_curBuffer;
		};
		bgfx::update(m_splatData[nextBuffer].m_vbh, 0, bgfx::makeRef(m_splatData[nextBuffer].m_buffer, vertexCount * sizeof(InstanceDataVertex), releaseFn, this));

		m_isSorting = false;
	}

	bool update() override
	{
		if (!entry::processEvents(m_width, m_height, m_debug, m_reset, &m_mouseState) )
		{
			int64_t now = bx::getHPCounter();
			static int64_t last = now;
			const int64_t frameTime = now - last;
			last = now;
			const double freq = double(bx::getHPFrequency() );
			const float deltaTime = float(frameTime/freq);

			imguiBeginFrame(m_mouseState.m_mx
				,  m_mouseState.m_my
				, (m_mouseState.m_buttons[entry::MouseButton::Left  ] ? IMGUI_MBUT_LEFT   : 0)
				| (m_mouseState.m_buttons[entry::MouseButton::Right ] ? IMGUI_MBUT_RIGHT  : 0)
				| (m_mouseState.m_buttons[entry::MouseButton::Middle] ? IMGUI_MBUT_MIDDLE : 0)
				,  m_mouseState.m_mz
				, uint16_t(m_width)
				, uint16_t(m_height)
				);

			showExampleDialog(this);

			imguiEndFrame();

			// Update camera.
			cameraUpdate(deltaTime / 30.0f, m_mouseState, ImGui::MouseOverArea() );

			if (!ImGui::MouseOverArea() )
			{
				if (m_mouseState.m_buttons[entry::MouseButton::Left])
				{
				}
			}

			// Set view 0 default viewport.
			bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height) );

			cameraGetViewMtx(m_viewMtx);
			bx::memCopy(m_curView, m_viewMtx, 16 * sizeof(float));

			bgfx::setState(
				BGFX_STATE_PT_TRISTRIP |
				BGFX_STATE_WRITE_RGB |
				BGFX_STATE_WRITE_A |
				BGFX_STATE_BLEND_EQUATION_ADD |
				BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_INV_DST_ALPHA, BGFX_STATE_BLEND_ONE) |
				0);

			bgfx::setUniform(u_focal, m_focal);
			bgfx::setViewTransform(0, m_viewMtx, m_projMtx);

			bgfx::setVertexBuffer(0, m_vbh);
			bgfx::setInstanceDataBuffer(m_splatData[m_curBuffer].m_vbh, 0, m_splatData[0].m_vertexCount);
			bgfx::submit(0, m_gaussianProgram);

			// Advance to next frame. Rendering thread will be kicked to
			// process submitted rendering primitives.
			bgfx::frame();

			if (bx::memCmp(m_curView, m_lastView, 16 * sizeof(float)) != 0) {
				float dot = m_lastView[2]  * m_curView[2] +
							m_lastView[6]  * m_curView[6] +
							m_lastView[10] * m_curView[10];
				if (abs(dot - 1.0) > 0.01 && !m_isSorting) {
					///*std::thread* th = */new std::thread(&ExampleGaussianSplatting::SortGaussians, this, m_viewMtx);
					SortGaussians(m_viewMtx);
					bx::memCopy(m_lastView, m_curView, 16 * sizeof(float));
					//m_curBuffer = 1 - m_curBuffer;
				}
			}

			return true;
		}

		return false;
	}

	bgfx::VertexBufferHandle m_vbh;
	std::vector<InstanceDataVertex>	m_splatFileData;
	GaussianSplatData	m_splatData[2];
	std::atomic<int>	m_curBuffer;
	std::atomic<bool>	m_isSorting = false;
	float bboxMin[3]{ FLT_MAX, FLT_MAX, FLT_MAX };
	float bboxMax[3]{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float m_range = 1.0f;

	bgfx::ProgramHandle m_gaussianProgram;
	bgfx::UniformHandle u_focal;

	float m_viewMtx[16];
	float m_projMtx[16];
	float m_curView[16];
	float m_lastView[16];
	float m_focal[4];

	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_debug;
	uint32_t m_reset;

	uint32_t m_oldWidth;
	uint32_t m_oldHeight;
	uint32_t m_oldReset;

	entry::MouseState m_mouseState;

	int64_t m_timeOffset;
};

} // namespace

ENTRY_IMPLEMENT_MAIN(
	ExampleGaussianSplatting
	, "50-gaussian-splatting"
	, "3D Gaussian Splatting example."
	, "https://bkaradzic.github.io/bgfx/examples.html#gaussian-splatting"
	);
