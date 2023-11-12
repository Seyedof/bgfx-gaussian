$input v_position, v_color

/*
 * Copyright 2023 Ali Seyedof. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#include "../common/common.sh"
//#include "common.sh"

void main()
{
	float A = -dot(v_position, v_position);
	if (A < -4.0) discard;
	float B = exp(A) * v_color.a;
	gl_FragColor = vec4(B * v_color.rgb, B);
}
