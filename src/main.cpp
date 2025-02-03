/**
 *
 *  @file      main.cpp
 *  @brief	   Application entry
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


/*------------------Includes---------------------*/
#include "main.h"
#include <chrono>
#include <unordered_map>
#include <vector>

/*------------Varible initialzation--------------*/
#define clock std::chrono::high_resolution_clock

HBITMAP h_bitmap                    = NULL;
HDC hdc_window                      = NULL;
BYTE* lpv_bits                      = NULL;
RECT window                         = { };
Camera cam                          = { };
int height                          = 0;
int width                           = 0;
bool camera_is_moving               = false;
bool drawing_frame                  = false;
float delta_time                    = 0.0f;
float move_speed                    = 0.1f;
float rotation_speed                = 2.0f;
long long nanoseconds_per_frame     = 33'333'333; /* 30 fps */
std::vector<Sphere> scene           = { };
std::vector<Light> lights           = { };
std::chrono::time_point clock_start = std::chrono::steady_clock( ).now( );
std::unordered_map<uint32_t, bool> key_states;


/*-------------Function Declaration--------------*/

void GameUpdate ( );

void RenderFrame ( HWND h_wnd );

int WINAPI WinMain ( _In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPSTR lpCmdLine,
                     _In_ int nCmdShow );

LRESULT CALLBACK WindowProc ( HWND hwnd,
                              UINT uMsg,
                              WPARAM wParam,
                              LPARAM lParam );

/*-------------Function Definitions--------------*/


/**
 * @brief Tells system to render a new frame
 *
 * @param h_wnd[in] - Handle to window
 */
void RenderFrame ( HWND h_wnd )
{
    Draw( &lpv_bits, width, height, cam, scene, lights );
    InvalidateRect( h_wnd, NULL, TRUE );
}

/**
 * @brief Updates camera movement, based on key_states
 */
void GameUpdate ( )
{
    // Camera updates should happen here
    if ( key_states['W'] )
    {
        cam.MoveForward( move_speed );
    }
    if ( key_states['A'] )
    {
        cam.MoveSideways( move_speed );
    }
    if ( key_states['S'] )
    {
        cam.MoveForward( -move_speed );
    }
    if ( key_states['D'] )
    {
        cam.MoveSideways( -move_speed );
    }
    if ( key_states['Q'] )
    {
        cam.yaw -= rotation_speed;
    }
    if ( key_states['E'] )
    {
        cam.yaw += rotation_speed;
    }
}

/**
 * @brief Application entrypoint.
 *
 * @param hInstance Handle to instance, base address of module memory
 * @param hPrevInstance Handle to previous instance - always NULL
 *		  if you need to detect if another exists, use CreateMutex
 *		  returns ERROR_ALREADY_EXISTS if theres already one named the same
 * @param lpCmdLine String to the command line for the application
 * @param nCmdShow Controls how the window is to be shown
 * @return
 */
int WINAPI WinMain ( _In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPSTR lpCmdLine,
                     _In_ int nCmdShow )
{
    // Register window class
    const wchar_t class_name[] = L"Window";


    WNDCLASS window_class = { };

    window_class.lpfnWndProc   = WindowProc;
    window_class.hInstance     = hInstance;
    window_class.lpszClassName = class_name;

    RegisterClass( &window_class );

    // Create window
    HWND hwnd = CreateWindowEx( 0,
                                class_name,
                                L"Win32 Raytracer",
                                WS_OVERLAPPEDWINDOW, // Style of window

                                CW_USEDEFAULT,
                                CW_USEDEFAULT,
                                700,
                                700,

                                NULL,
                                NULL,
                                hInstance,
                                NULL );

    if ( hwnd == NULL )
    {
        return 0;
    }
    CreateScene( scene, lights );
    GetClientRect( hwnd, &window );
    Init( &lpv_bits, &h_bitmap, &window );
    ShowWindow( hwnd, nCmdShow );
    UpdateWindow( hwnd );

    MSG msg = { };

    auto delta_epoch = std::chrono::steady_clock::now( );
    auto cached_time = std::chrono::steady_clock::now( );

    while ( msg.message != WM_QUIT )
    {
        if ( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            // game loop
            auto now = std::chrono::steady_clock::now( );
            if ( ( now - clock_start ).count( ) >= nanoseconds_per_frame )
            {
                clock_start = now;
                delta_time  = 1.0f; // Use this to scale camera speed
                GameUpdate( );
                drawing_frame = true;
            }
            if ( drawing_frame )
            {
                RenderFrame( hwnd );
                drawing_frame = false;
            }
        }
    }
    UnregisterClass( L"Win32 Raytracer", window_class.hInstance );
    return 0;
}

/**
 * Called from message dispatch, determines response to messages.
 *
 * @param hwnd handle to the window
 * @param uMsg message code; e.g. WM_KEYDOWN
 * @param wParam data pertaining to message e.g. which key pressen on
 * WM_KEYDOWN
 * @param lParam data pertaining to message if neeeded
 * @return
 */
LRESULT CALLBACK WindowProc ( HWND hwnd,
                              UINT uMsg,
                              WPARAM wParam,
                              LPARAM lParam )
{
    width  = window.right;
    height = window.bottom;

    switch ( uMsg )
    {
        case WM_CREATE:
        {
            break;
        }
        case WM_DESTROY:
        {
            // Cleanup
            if ( h_bitmap )
            {
                DeleteObject( h_bitmap );
            }
            PostQuitMessage( 0 );

            break;
        }

        case WM_PAINT:
        {
            PAINTSTRUCT ps;

            GetClientRect( hwnd, &window ); // Use client area dimensions

            width   = window.right;
            height  = window.bottom;
            HDC hdc = BeginPaint( hwnd, &ps );

            HDC hdcMem        = CreateCompatibleDC( hdc );
            HGDIOBJ oldBitmap = SelectObject( hdcMem, h_bitmap );

            BitBlt( hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY );

            SelectObject( hdcMem, oldBitmap );
            DeleteDC( hdcMem );

            // All painting occurs here, between BeginPaint and EndPaint.

            EndPaint( hwnd, &ps );

            break;
        }
        case WM_MOVE:
        {
            // TODO: do we want to do something on move?
        }
        case WM_KEYUP:
        {
            key_states[static_cast<uint32_t>( wParam )] = false;
            break;
        }
        case WM_KEYDOWN:
        {
            key_states[static_cast<uint32_t>( wParam )] = true;
            break;
        }
        default:
        {
            return DefWindowProc( hwnd, uMsg, wParam, lParam );
        }
    }
    return DefWindowProc( hwnd, uMsg, wParam, lParam );
}
