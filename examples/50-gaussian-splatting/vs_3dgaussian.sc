$input a_position, a_color0, i_data0, i_data1, i_data2, i_data3
$output v_position, v_color

/*
 * Copyright 2023 Ali Seyedof. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#include "../common/common.sh"
//#include "common.sh"

uniform vec4 u_focal;

#define color	i_data0
#define center	i_data1
#define covA	i_data2
#define covB	i_data3

void main()
{
	vec4 camspace = mul(u_view, vec4(center.xyz, 1));
	vec4 pos2d = mul(u_proj, camspace);

	float bounds = 1.2 * pos2d.w;
	if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds	|| pos2d.y < -bounds || pos2d.y > bounds) {
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
	}
	else {
		mat3 Vrk = mat3(
			covA.x, covA.y, covA.z,
			covA.y, covB.x, covB.y,
			covA.z, covB.y, covB.z
		);

		mat3 J = mat3(
			u_focal.x,	0.0,		-(u_focal.x * camspace.x) / camspace.z,
			0.0,		-u_focal.y,	(u_focal.y * camspace.y) / camspace.z,
			0.0,		0.0,		0.0
		) / camspace.z;

#if BGFX_SHADER_LANGUAGE_GLSL
		mat3 W = transpose(mat3(u_view));
#else
		mat3 W = transpose((mat3)u_view);
#endif
		mat3 T = mul(W, J);
		mat3 cov = transpose(T) * Vrk * T;

		vec2 vCenter = pos2d.xy / pos2d.w;
		vCenter.y = -vCenter.y;

		float diagonal1 = cov[0][0] + 0.3;
		float offDiagonal = cov[0][1];
		float diagonal2 = cov[1][1] + 0.3;

		float mid = 0.5 * (diagonal1 + diagonal2);
		float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
		float lambda1 = mid + radius;
		float lambda2 = max(mid - radius, 0.1);
		vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
		vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
		vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

		v_color = color;
		v_position = a_position;

		vec2 viewSize = vec2(u_viewRect.z, u_viewRect.w);

		gl_Position = vec4(
			vCenter +
			v_position.x * v1 / viewSize * 2.0 +
			v_position.y * v2 / viewSize * 2.0,
			0.0, 1.0);
	}
}
