/**
 *
 *  @file      raytrace.cpp
 *  @brief     Declaration of core raytracing functionality
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


/*------------------Libraries---------------------*/
//#include "raytrace.h"
#include "raytrace.cuh"
/*------------Varible initialzation---------------*/

Sphere scene[4] = {};
Light lights[3] = {};
//
///*------------Funcition Defenitions---------------*/
//
//Vect3D ReflectRay(const Vect3D &R, const Vect3D &N)
//{
//	return ((N*(N.dot(R))) * 2) - R;
//}
//
//void CreateScene()
//{
//	scene[0].center = Vect3D(0, -1, 3);
//	scene[0].radius = 1;
//	scene[0].color = RGB(255, 0, 0);
//	scene[0].specularity = 500;
//	scene[0].reflective = 0.2;
//	scene[0].sRadius = scene[0].radius * scene[0].radius;
//
//	scene[1].center = Vect3D(2, 0, 4);
//	scene[1].radius = 1;
//	scene[1].color = RGB(0, 0, 255);
//	scene[1].specularity = 500;
//	scene[1].reflective = 0.3;
//	scene[1].sRadius = scene[1].radius * scene[1].radius;
//
//	scene[2].center = Vect3D(-2, 0, 4);
//	scene[2].radius = 1;
//	scene[2].color = RGB(0, 255, 0);
//	scene[2].specularity = 10;
//	scene[2].reflective = 0.4;
//	scene[2].sRadius = scene[2].radius * scene[2].radius;
//
//	scene[3].center = Vect3D(0, -5001, 0);
//	scene[3].radius = 5000;
//	scene[3].color = RGB(255, 255, 0);
//	scene[3].specularity = 1000;
//	scene[3].reflective = 0.5;
//	scene[3].sRadius = scene[3].radius * scene[3].radius;
//
//	lights[0].type = lights->AMBIENT;
//	lights[0].intensity = 0.2;
//	//lights[0].pos = { 0,0,0 }; //prettysure this is useless
//
//	lights[1].type = lights->POINT;
//	lights[1].intensity = 0.6;
//	lights[1].pos = { 2, 1, 0 };
//
//	lights[2].type = lights->DIRECTIONAL;
//	lights[2].intensity = 0.2;
//	lights[2].pos = { 1, 4, 4 };
//}
//
//double CalcLight(const Vect3D &P, const Vect3D &N, const Vect3D &V, const int &s)
//{
//	double intensity = 0.0;
//	double t_max = 0;
//	Vect3D L = {};
//	Vect3D R = {};
//	for (int i = 0; i < sizeof(lights) / sizeof(Light); i++)
//	{
//		if (lights[i].type == lights->AMBIENT)
//		{
//			intensity += lights[i].intensity;
//		}
//		else
//		{
//			if (lights[i].type == lights->POINT)
//			{
//				L = (lights[i].pos - P);
//				t_max = 1;
//			}
//			else
//			{
//				L = lights[i].pos;
//				t_max = INFINITY;
//			}
//			L = L;
//			auto [shadow_sphere, shadow_t] = ClosestIntersection(P, L, 0.00001, t_max);
//			if (shadow_sphere != NULL)
//			{
//				continue;
//			}
//
//			double n_dot_l = N.dot(L);
//			if (n_dot_l > 0)
//			{
//				intensity += lights[i].intensity * n_dot_l / (N.len() * L.len());
//			}
//
//			if (s != -1)
//			{
//				R = ReflectRay(L, N);
//				double r_dot_v = R.dot(V);
//
//				if (r_dot_v > 0)
//				{
//					intensity += lights[i].intensity * pow(r_dot_v/(R.len() * (V.len())), s);
//				}
//
//			}
//		}
//	}
//	return intensity;
//}
//
//void Init(BYTE** pLpvBits, const RECT* window, HBITMAP* pHBitmap)
//{
//	CreateScene();
//	int width = (*window).right;
//	int height = (*window).bottom;
//
//	BITMAPINFO bmi = {};
//	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
//	bmi.bmiHeader.biWidth = width;
//	bmi.bmiHeader.biHeight = -height; // Negative to have a top-down DIB
//	bmi.bmiHeader.biPlanes = 1;
//	bmi.bmiHeader.biBitCount = 32;
//	bmi.bmiHeader.biCompression = BI_RGB;
//
//	// Create the DIB section and obtain a pointer to the pixel buffer
//	*pHBitmap = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, (void**)&(*pLpvBits), NULL, 0);
//
//	if (!(*pLpvBits) || !(*pHBitmap)) {
//		MessageBox(NULL, L"Could not allocate memory for bitmap", L"Error", MB_OK | MB_ICONERROR);
//		exit(1);
//	}
//
//	// Initialize all pixels to black
//	memset(*pLpvBits, 0, width * height * 4);
//
//
//}
//
//double IntersectRaySphere(const Vect3D &O, const Vect3D &D, const Sphere &sphere, const double &dDot)
//{
//	Vect3D CO = {};
//	CO = O - sphere.center;
//
//	double a = dDot;
//	double b = 2 * CO.dot(D);
//	double c = CO.dot(CO) - sphere.sRadius;
//
//	double discr = b * b - 4 * a * c;
//
//	if (discr < 0)
//	{
//		return INFINITY;
//	}
//	else if (discr == 0)
//	{
//		return -b / (2 * a);
//	}
//
//	double t = (-b - sqrt(discr)) / (2 * a);		// Minimize compute only go for 1 root;
//
//	return t;
//}
//
//Vect3D CanvasToViewport(const int &x, const int &y, const int &width, const int &height)
//{
//	// for simplicity : Vw = Vh = d = 1    approx 53 fov
//	double aspectRatio = static_cast<double>(width) / height;
//
//	// Map x and y to the viewport, adjusting by aspect ratio
//	double fovMod = 1;
//	double viewportX = (x - width / 2.0) * ((1.0 * fovMod) / width) * aspectRatio;
//	double viewportY = -(y - height / 2.0) * ((1.0 * fovMod) / height); // Flip Y to match 3D space orientation
//
//	return Vect3D(viewportX, viewportY, 1);  // Z=1 for perspective projection
//}
//
//Intersection ClosestIntersection(const Vect3D &O, const Vect3D &D, const double &t_min, const double &t_max)
//{
//	double closest_t = INFINITY;
//	Sphere *closest_sphere = NULL;
//	double d_dot_d = D.dot(D);		// Cache immutable value
//	for (auto &x : scene)
//	{
//		double t = IntersectRaySphere(O, D, x, d_dot_d);
//
//		if (IsInBounds(t, t_min, t_max) && t < closest_t)
//		{
//			closest_t = t;
//			closest_sphere = const_cast<Sphere*>(&x);
//		}
//	}
//	return Intersection(closest_sphere, closest_t);
//}
//
//COLORREF TraceRay(const Vect3D &O, const Vect3D &D, const double &t_min, const double &t_max, const int &recursionDepth)
//{
//	Vect3D N = {};
//	Vect3D P = {};
//	Vect3D R = {};
//
//	auto [closest_sphere, closest_t] = ClosestIntersection(O, D, t_min, t_max);
//
//	if (closest_sphere == NULL)
//	{
//		return RGB(0, 0, 0);
//	}
//	
//	P = O + (D * closest_t);
//	N = (P - closest_sphere->center).norm();
//
//	double res = CalcLight(P, N, D.invert(), closest_sphere->specularity);
//	int r = (int)round(GetRValue(closest_sphere->color) * res);
//	int g = (int)round(GetGValue(closest_sphere->color) * res);
//	int b = (int)round(GetBValue(closest_sphere->color) * res);
//
//	double refl = closest_sphere->reflective;
// 
//	if (recursionDepth <= 0 || refl <= 0) 
//	{
//		return RGB(max(0, min(255, r)),
//				   max(0, min(255, g)),
//				   max(0, min(255, b)));
//	}
//
//
//	R = ReflectRay(D.invert(), N);
//	COLORREF reflectedColor = TraceRay(P, R, t_min, t_max, recursionDepth - 1);
//
//	int reflected_r = (int)round(GetRValue(reflectedColor)) * refl;
//	int reflected_g = (int)round(GetGValue(reflectedColor)) * refl;
//	int reflected_b = (int)round(GetBValue(reflectedColor)) * refl;
//
//	
//	return RGB(max(0, min(255, r * (1 - refl) + reflected_r)),
//			   max(0, min(255, g * (1 - refl) + reflected_g)),
//			   max(0, min(255, b * (1 - refl) + reflected_b)));
//
//}


void Draw(BYTE** pLpvBits, const int &width, const int &height, Camera &cam)
{
	//Vect3D D = {};
	//Vect3D N = {};
	//Vect3D P = {};
	//const Vect3D O = { 0,0,0 };
	//double t_min = 0.0001;
	//double t_max = INFINITY;
	//int recursionDepth = 2;

	//for (int x = 0; (x < (width)); ++x)
	//{
	//	for (int y = 0; (y < (height)); ++y)
	//	{
	//		D = CanvasToViewport(x, y, width, height).norm();
	//		D = cam.ApplyCameraRotation(D, cam).norm();
	//		COLORREF color = TraceRay(cam.position, D, t_min, t_max, recursionDepth);
	//		D = D.norm();

	//		int offset = (y * width + x) * 4;
	//		if (offset >= 0 && offset < width * height * 4 - 4) {
	//			(*pLpvBits)[offset + 0] = (int)GetBValue(color);
	//			(*pLpvBits)[offset + 1] = (int)GetGValue(color);
	//			(*pLpvBits)[offset + 2] = (int)GetRValue(color);
	//			(*pLpvBits)[offset + 3] = 255;
	//		}

	//	}
	//}
	Draw_Caller(pLpvBits, cam, scene, lights);

}

