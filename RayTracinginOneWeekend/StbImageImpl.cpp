// === stb_image 구현부 전용 호스트 번역 단위 ===
//
// stb_image.h의 실제 구현(STB_IMAGE_IMPLEMENTATION)은 약 5000줄의 순수
// 호스트 C 코드다. 이를 kernel.cu(.cu) 안에 넣으면 nvcc의 디바이스 분리
// 프런트엔드(cudafe++)가 이 거대한 코드를 파싱하다가 죽는다(ACCESS_VIOLATION).
//
// 그래서 구현부는 이 .cpp 파일에서만 빌드한다. 이 파일은 nvcc가 아니라
// 호스트 컴파일러(cl)로 컴파일되며, RtwImage.h(헤더, 선언부만 포함)에서
// 호출하는 stbi_loadf/stbi_image_free 등의 실제 정의를 제공한다.
//
// 즉, 역할 분담:
//   - StbImageImpl.cpp : stb 구현부(이 파일, 호스트 전용)
//   - RtwImage.h       : stb 선언부 + 디바이스 업로드 로직(.cu에서 include)

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "external/stb_image.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif
