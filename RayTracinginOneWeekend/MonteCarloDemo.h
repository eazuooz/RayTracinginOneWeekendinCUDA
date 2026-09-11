#pragma once
#ifndef MONTE_CARLO_DEMO_H
#define MONTE_CARLO_DEMO_H

// === Ray Tracing: The Rest of Your Life — 몬테카를로 데모 (GPU) ===
//
// 원서 3권 앞부분의 "레이트레이서가 아닌 작은 수치 실험 프로그램"(pi.cc,
// integrate_x_sq.cc, ...)을 GPU 커널로 옮긴 모음. 렌더러와 독립된 번역 단위
// (MonteCarloDemo.cu)에 두고, main()에서 --demo <이름> 으로 실행한다.
// 호스트 전용 선언만 두어 kernel.cu 쪽 컴파일에 영향을 주지 않는다.

// 이름에 해당하는 데모를 실행한다. 이름을 알아들었으면 true.
bool RunMonteCarloDemo(const char* name);

// 사용 가능한 데모 목록을 stderr로 출력한다.
void PrintMonteCarloDemoList();

#endif
