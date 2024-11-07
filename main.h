#pragma once

/*------------------Libraries---------------------*/
#include "Windows.h"
#include <vector>
#include <math.h>
#include <iostream>
/*----------------My Libraries---------------------*/
#include "raytrace.h"


/*-------------Function Declarations-------------*/


int WINAPI WinMain(_In_ HINSTANCE hInstance,
					_In_opt_ HINSTANCE hPrevInstance,
					_In_ LPSTR lpCmdLine,
					_In_ int nCmdShow);


/**
 * .
 */
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
/*---------------------------------------*/
/*---------------------------------------*/
/*---------------------------------------*/


