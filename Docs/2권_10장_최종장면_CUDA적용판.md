# A Scene Testing All New Features (모든 새로운 기능을 테스트하는 장면) — CUDA 적용판

> *Ray Tracing: The Next Week* 마지막 장(A Scene Testing All New Features)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 옮긴 문서.
> 설명 본문은 **원서 원문을 그대로** 싣고(영문 + 국문), 코드만 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 실제로 VS2022 + CUDA 12.9로 빌드·실행해 최종 장면을 확인했다.

---

## 원문 (Original Text)

> Let’s put it all together, with a big thin mist covering everything, and a blue subsurface reflection sphere (we didn’t implement that explicitly, but a volume inside a dielectric is what a subsurface material is). The biggest limitation left in the renderer is no shadow rays, but that is why we get caustics and subsurface for free. It’s a double-edged design decision.

> 이제 모든 요소를 통합해 보겠습니다. 전체를 덮는 크고 얇은 안개와 청색의 표면 하부 반사 구 (이를 명시적으로 구현하지는 않았지만, 유전체 내부의 부피가 바로 표면 하부 재질입니다) 를 추가합니다. 현재 렌더러에서 남은 가장 큰 제한 사항은 섀도 레이를 지원하지 않는다는 점이지만, 이로 인해 코스틱스와 표면 하부 효과를 무료로 얻을 수 있습니다. 이는 양날의 검과 같은 설계 결정입니다.

> Also note that we'll parameterize this final scene to support a lower quality render for quick testing.

> 또한 빠른 테스트를 위해 낮은 품질의 렌더링을 지원하도록 이 최종 장면을 매개변수화할 것임을 유의하시기 바랍니다.

---

## 원서 코드 (Listing 74)

원서의 `final_scene()` 원본은 다음과 같다(참고용 원문).

```cpp
void final_scene(int image_width, int samples_per_pixel, int max_depth) {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;
            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list world;
    world.add(make_shared<bvh_node>(boxes1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    world.add(make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

    world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(
        point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
    world.add(boundary);
    world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<sphere>(point3(0,0,0), 5000, make_shared<dielectric>(1.5));
    world.add(make_shared<constant_medium>(boundary, .0001, color(1,1,1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    world.add(make_shared<sphere>(point3(400,200,400), 100, emat));
    auto pertext = make_shared<noise_texture>(0.2);
    world.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
    }

    world.add(make_shared<translate>(
        make_shared<rotate_y>(
            make_shared<bvh_node>(boxes2), 15),
            vec3(-100,270,395)));

    camera cam;
    cam.aspect_ratio      = 1.0;
    cam.image_width       = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth         = max_depth;
    cam.background        = color(0,0,0);
    cam.vfov     = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);
    cam.defocus_angle = 0;
    cam.render(world);
}

int main() {
    switch (9) {
        case 1:  bouncing_spheres();          break;
        ...
        case 8:  cornell_smoke();             break;
        case 9:  final_scene(800, 10000, 40); break;
        default: final_scene(400,   250,  4); break;
    }
}
```

***Listing 74:** [main.cc] Final scene*

> Running it with 10,000 rays per pixel (sweet dreams) yields:
>
> 픽셀당 10,000 개의 광선으로 실행하면 (행운을 빕니다) 다음과 같은 결과가 나옵니다:

![Image 23: Final scene](https://raytracing.github.io/images/img-2.22-book2-final.jpg)

---

## CUDA 적용판 (`kernel.cu`, `sceneId == 9`)

원서 `final_scene()`를 우리 `CreateWorld` 커널의 `sceneId == 9` 분기로 옮겼다. 기능·배치·좌표·색은 원서와 동일하고, 달라진 부분은 **모두 아래 "CUDA 적용 메모"** 에 정리했다.

```cpp
else  // sceneId == 9, final_scene
{
    // 바닥: 20x20 = 400개, 높이가 랜덤인 상자 (원서 boxes1)
    Material* ground = new Lambertian(Color(0.48, 0.83, 0.53));
    const int boxesPerSide = 20;
    for (int bi = 0; bi < boxesPerSide; bi++)
        for (int bj = 0; bj < boxesPerSide; bj++)
        {
            double w = 100.0;
            double x0 = -1000.0 + bi * w;
            double z0 = -1000.0 + bj * w;
            double x1 = x0 + w;
            double y1 = 1.0 + 100.0 * RND;      // random_double(1,101)
            double z1 = z0 + w;
            list[i++] = MakeBox(Point3(x0, 0.0, z0), Point3(x1, y1, z1), ground);
        }

    // 광원
    Material* light = new DiffuseLight(Color(7, 7, 7));
    list[i++] = new Quad(Vector3(123, 554, 147), Vector3(300, 0, 0), Vector3(0, 0, 265), light);

    // 모션 블러 구 (center1 → center2)
    Material* sphereMaterial = new Lambertian(Color(0.7, 0.3, 0.1));
    list[i++] = new MovingSphere(
        Point3(400, 400, 200), Point3(430, 400, 200), 0.0, 1.0, 50.0, sphereMaterial);

    // 유리 구 + 금속 구
    list[i++] = new Sphere(Point3(260, 150, 45), 50.0, new Dielectric(1.5));
    list[i++] = new Sphere(Point3(0, 150, 145), 50.0, new Metal(Color(0.8, 0.8, 0.9), 1.0));

    // 청색 표면하부 산란 구: 유리 경계 + 내부 볼륨 (구를 2개 만든다 — CUDA 메모 참고)
    list[i++] = new Sphere(Point3(360, 150, 145), 70.0, new Dielectric(1.5));
    Hittable* blueBoundary = new Sphere(Point3(360, 150, 145), 70.0, new Dielectric(1.5));
    list[i++] = new ConstantMedium(blueBoundary, 0.2, Color(0.2, 0.4, 0.9));

    // 전체를 덮는 얇은 안개: 거대한(반지름 5000) 유리 구 경계 + 매우 옅은 볼륨
    Hittable* mistBoundary = new Sphere(Point3(0, 0, 0), 5000.0, new Dielectric(1.5));
    list[i++] = new ConstantMedium(mistBoundary, 0.0001, Color(1, 1, 1));

    // 지구 이미지 텍스처 구 (earthmap.jpg)
    Texture* earthTex = new ImageTexture(earthData, earthW, earthH);
    list[i++] = new Sphere(Point3(400, 200, 400), 100.0, new Lambertian(earthTex));

    // 펄린 노이즈 구
    Texture* pertext = new NoiseTexture(0.2, &localRandState);
    list[i++] = new Sphere(Point3(220, 280, 300), 80.0, new Lambertian(pertext));

    // 1000개의 작은 흰 구 클러스터 → 15° 회전 + 이동
    Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
    const int ns = 1000;
    Hittable** boxes2 = new Hittable*[ns];
    for (int s = 0; s < ns; s++)
    {
        Point3 c(165.0 * RND, 165.0 * RND, 165.0 * RND);   // point3::random(0,165)
        boxes2[s] = new Sphere(c, 10.0, white);
    }
    Hittable* cluster = new HittableList(boxes2, ns, true);   // 1000개 구를 소유
    cluster = new RotateY(cluster, 15.0);
    cluster = new Translate(cluster, Vector3(-100, 270, 395));
    list[i++] = cluster;

    background = Color(0, 0, 0);
    lookfrom = Vector3(478, 278, -600); lookat = Vector3(278, 278, 0); vfov = 40.0;
    shutterOpen = 0.0; shutterClose = 1.0;    // 모션 블러 셔터 구간
}
```

---

## CUDA 적용 메모

원서(CPU, `shared_ptr` + `hittable_list` + STL)와 달라진 점만 모았다.

1. **`random_double` / `point3::random` → `RND` 매크로**
   `final_scene`의 무작위 값들은 호스트 `random_double`을 쓴다. 우리는 `CreateWorld` 안에서 단일 스레드 cuRAND(`localRandState`)를 쓰는 `#define RND (curand_uniform(&localRandState))`로 바꾼다.
   - `random_double(1,101)` → `1.0 + 100.0 * RND`
   - `point3::random(0,165)` → `Point3(165*RND, 165*RND, 165*RND)`

2. **`box()` / `bvh_node` / 인스턴스 → 우리 프리미티브**
   `box(...)` → `MakeBox(...)`([[2권_8장_인스턴스]]), `rotate_y`/`translate` → `RotateY`/`Translate`, 이동 구는 `MovingSphere`. 바닥의 400개 상자는 원서가 `bvh_node(boxes1)` 하나로 묶지만, 우리는 **400개를 그대로 `list[]`에 넣고 메인 BVH가 함께 묶는다**(별도 sub-BVH 불필요, 오히려 통합되어 더 단순하다).

3. **⚠️ 청색 표면하부 구 — 구를 2개 만든다 (더블 프리 회피)**
   원서는 **같은 `boundary` 구 포인터**를 "보이는 유리"와 "볼륨의 경계" 양쪽에 공유한다(`shared_ptr`라 안전).
   ```cpp
   auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
   world.add(boundary);                                          // 보이는 유리
   world.add(make_shared<constant_medium>(boundary, 0.2, ...));  // 같은 포인터 공유
   ```
   우리의 raw 포인터 모델에서는 `ConstantMedium`이 경계를 `delete`하므로, 같은 포인터를 `list[]`에도 넣으면 **더블 프리**가 난다. 그래서 **좌표·반지름이 같은 구를 2개** 만든다(하나는 보이는 유리, 하나는 볼륨 전용 경계). 화면 결과는 동일하다.

4. **1000개 구 클러스터 — sub-BVH 대신 "소유 HittableList"**
   원서는 `translate(rotate_y(bvh_node(boxes2), 15), ...)`로 1000개 구의 **sub-BVH**를 만든다. 그런데 우리 `Translate`/`RotateY`는 소멸자에서 **자식을 `delete`** 한다([[2권_8장_인스턴스]]). 이게 BVH 노드 레지스트리 해제(`bvhNodes[]`)와 충돌해 **더블 프리**가 난다. 그래서 클러스터는 **6개 면 상자(MakeBox)와 같은 "소유 HittableList"(`bOwns=true`)** 로 묶는다.
   - 해제: `FreeWorld`가 최상위 `Translate`만 `delete` → `RotateY → HittableList(소유) → 1000 Sphere` 연쇄 해제(누수/더블 프리 없음).
   - 성능: 메인 BVH가 이 그룹의 경계 상자로 컬링하므로, **그룹 bbox에 들어온 레이만 1000개를 선형 검사**한다(아래 성능 메모 참고).

5. **이미지 텍스처(earthmap.jpg)** 는 호스트(`RtwImage`)가 디코딩해 디바이스로 올린 버퍼를 `ImageTexture`가 인덱싱한다([[2권_4장_텍스처매핑]]). `main()`의 업로드 조건을 `sceneId == 2 || sceneId == 9`로 넓혔다. 파일이 없으면 청록색으로 표시된다.

6. **거대한 안개 경계 구(반지름 5000)** 는 원점 중심이라 **카메라가 그 내부**에 있다. `ConstantMedium::Hit`의 "레이 시작점이 매질 내부" 경로(`rec1.t < 0 → 0`)가 이를 처리한다([[2권_9장_볼륨]]). 이 볼륨의 경계 상자는 장면 전체를 덮어 BVH로 컬링되지 않으므로 **모든 레이가 한 번씩** 이 매질을 검사한다(연산은 가볍다).

7. **품질 매개변수화 → `numSamples`**
   원서는 `final_scene(image_width, samples_per_pixel, max_depth)`로 품질을 인자화한다(고품질 `(800, 10000, 40)`, 빠른 테스트 `(400, 250, 4)`). 우리는 해상도(1440×720)·최대 바운스(RayColor 루프 50)는 고정이고, **샘플 수만** `main()`에서 장면별로 정한다. 최종 장면(9)은 무거워 **100 spp**로 둔다(원서의 10000은 GPU에서도 매우 오래 걸린다).

> 🔧 **성능 메모 (측정값)**: 1440×720 / **100 spp** 기준 약 **256초**(VS2022 + CUDA 12.9, Debug). 1000개 구 클러스터의 선형 검사와 볼륨 산란이 주요 비용이다. 샘플을 올리면 노이즈가 줄지만 시간이 비례해 늘어난다.

---

## 결과 & 검증

VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**(1440×720, 100 spp, 약 256초). 원서 Image 23과 동일한 구도로, 다음 요소가 모두 한 장면에 나타나는 것을 확인했다:

- 높이가 랜덤인 **녹색 바닥 상자 400개**(`MakeBox`)
- 천장 **사각형 광원**(`Quad` + `DiffuseLight`)
- **모션 블러 구**(`MovingSphere`)
- **유리/금속 구**(`Dielectric`/`Metal`)
- **청색 표면하부 산란 구**(유리 경계 + `ConstantMedium`)
- 전체를 덮는 **얇은 안개**(반지름 5000 볼륨)
- **지구 이미지 텍스처 구**(`ImageTexture`) + **펄린 노이즈 구**(`NoiseTexture`)
- **1000개 작은 구 클러스터**(소유 `HittableList` → `RotateY` → `Translate`)

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 74 | `kernel.cu` (scene 9) | `final_scene`을 `CreateWorld`의 `sceneId == 9`로 이식 |
| — | `kernel.cu` (main) | `sceneId = 9` 기본값, `numSamples`(9→100), earthmap 업로드 조건에 9 추가, `maxHittables`=512 |

> 이 장은 **새 클래스 없이** 앞 장들(텍스처·펄린·사각형·조명·인스턴스·볼륨·모션 블러)에서 만든 프리미티브를 한 장면에 모은 통합 장면이다. CUDA 적용에서 새로 신경 쓸 점은 **포인터 공유로 인한 더블 프리**(청색 구·1000 구 클러스터)뿐이며, 둘 다 "구를 2개 만들기 / 소유 HittableList로 묶기"로 해결했다.

### CUDA 적용에서 꼭 기억할 3가지

1. **무작위 값은 `RND` 매크로로**: `random_double`/`point3::random`을 `CreateWorld`의 단일 스레드 cuRAND(`RND`)로 바꾼다.
2. **포인터 공유 = 더블 프리 주의**: 청색 표면하부 구는 경계를 공유하지 말고 **2개 만들고**, 1000 구 클러스터는 sub-BVH 대신 **소유 HittableList**로 묶는다.
3. **품질은 `numSamples`로 매개변수화**: 최종 장면은 무거우니 100 spp로 시작하고, 노이즈가 거슬리면 샘플을 올린다(원서 10000은 GPU에서도 매우 느리다).
```
