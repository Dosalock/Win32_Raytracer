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
#include "raytrace.h"

/*------------Varible initialzation---------------*/


/*------------Funcition Defenitions---------------*/

/**
 * @brief Checks if a root is within limits and is a better match than
 * previously cached
 * @param root - Root to check
 * @param t_min - Minimum acceptable value for root
 * @param t_max - Maximum acceptable value for root
 * @param closest_t - Current best root
 * @return
 */
bool IsBetterRoot ( _In_ const float root,
                    _In_ const float t_min,
                    _In_ const float t_max,
                    _In_ const float closest_t )
{
    return IsInBounds( root, t_min, t_max ) && root < closest_t;
}

Vect3D ReflectRay ( _In_ const Vect3D ray_to_reflect,
                    _In_ const Vect3D sphere_normal )
{
    return ( ( sphere_normal * ( sphere_normal.dot( ray_to_reflect ) ) ) * 2 )
           - ray_to_reflect;
}

void CreateScene ( _Out_ std::vector<Sphere> &scene,
                   _Out_ std::vector<Light> &lights )
{
    Sphere sphere = { };

    sphere.center         = Vect3D( 0, -1, 3 );
    sphere.radius         = 1.0f;
    sphere.color          = WideColor( 255, 0, 0 );
    sphere.specularity    = 500;
    sphere.reflective     = 0.2f;
    sphere.raidus_squared = sphere.radius * sphere.radius;
    scene.push_back( sphere );

    sphere.center         = Vect3D( 2, 0, 4 );
    sphere.radius         = 1;
    sphere.color          = WideColor( 0, 0, 255 );
    sphere.specularity    = 500;
    sphere.reflective     = 0.3;
    sphere.raidus_squared = sphere.radius * sphere.radius;
    scene.push_back( sphere );

    sphere.center         = Vect3D( -2, 0, 4 );
    sphere.radius         = 1;
    sphere.color          = WideColor( 0, 255, 0 );
    sphere.specularity    = 10;
    sphere.reflective     = 0.4;
    sphere.raidus_squared = sphere.radius * sphere.radius;
    scene.push_back( sphere );

    sphere.center         = Vect3D( 0, -5001, 0 );
    sphere.radius         = 5000;
    sphere.color          = WideColor( 255, 255, 0 );
    sphere.specularity    = 1000;
    sphere.reflective     = 0.5;
    sphere.raidus_squared = sphere.radius * sphere.radius;
    scene.push_back( sphere );

    Light light     = { };
    light.type      = light.AMBIENT;
    light.intensity = 0.2f;
    light.pos       = { 0, 0, 0 }; // prettysure this is useless
    lights.push_back( light );

    light.type      = light.POINT;
    light.intensity = 0.6f;
    light.pos       = { 2, 1, 0 };
    lights.push_back( light );

    light.type      = light.DIRECTIONAL;
    light.intensity = 0.2f;
    light.pos       = { 1, 4, 4 };
    lights.push_back( light );
}

float CalcLight ( _In_ const Vect3D intersection_point,
                  _In_ const Vect3D normalized_sphere_vector,
                  _In_ const Vect3D point_to_camera,
                  _In_ const uint32_t sphere_specularity,
                  _In_ std::vector<Sphere> &scene,
                  _In_ std::vector<Light> &lights )
{
    float intensity       = 0.0;
    float t_max           = 0;
    Vect3D light_position = { };

    for ( auto &light : lights )
    {
        if ( light.type == Light::AMBIENT )
        {
            intensity += light.intensity;
            if ( intensity >= 1.0f )
            {
                return 1.0f;
            }
        }
        else
        {
            if ( light.type == Light::POINT )
            {
                light_position = ( light.pos - intersection_point );
                t_max          = 1;
            }
            else
            {
                light_position = light.pos;
                t_max          = INFINITY;
            }

            float t_min = 0.00001f;

            auto [shadow_sphere, shadow_point_t] =
                ClosestIntersection( intersection_point,
                                     light_position,
                                     t_min,
                                     t_max,
                                     scene );
            if ( shadow_sphere != NULL )
            {
                continue;
            }

            float sphere_light_alignment =
                normalized_sphere_vector.dot( light_position );

            /* If < it is behind */
            if ( sphere_light_alignment > 0 )
            {
                intensity += light.intensity * sphere_light_alignment
                             / ( normalized_sphere_vector.len( )
                                 * light_position.len( ) );
            }

            if ( sphere_specularity != -1 )
            {
                Vect3D sphere_to_light =
                    ReflectRay( light_position, normalized_sphere_vector );

                float vector_alignment_to_camera =
                    sphere_to_light.dot( point_to_camera );

                /* If < 0 the object is behind the camera */
                if ( vector_alignment_to_camera > 0 )
                {
                    intensity += light.intensity
                                 * powf( vector_alignment_to_camera
                                             / ( sphere_to_light.len( )
                                                 * ( point_to_camera.len( ) ) ),
                                         sphere_specularity );
                }
            }
        }
    }
    return intensity;
}

void Init ( _Inout_ BYTE **pLpvBits,
            _Inout_ HBITMAP *pHBitmap,
            _In_ const RECT *window )
{
    int width  = ( *window ).right;
    int height = ( *window ).bottom;

    BITMAPINFO bmi              = { };
    bmi.bmiHeader.biSize        = sizeof( BITMAPINFOHEADER );
    bmi.bmiHeader.biWidth       = width;
    bmi.bmiHeader.biHeight      = -height; // Negative to have a top-down DIB
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    // Create the DIB section and obtain a pointer to the pixel buffer
    *pHBitmap = CreateDIBSection( NULL,
                                  &bmi,
                                  DIB_RGB_COLORS,
                                  ( void ** )&( *pLpvBits ),
                                  NULL,
                                  0 );

    if ( !( *pLpvBits ) || !( *pHBitmap ) )
    {
        MessageBox( NULL,
                    L"Could not allocate memory for bitmap",
                    L"Error",
                    MB_OK | MB_ICONERROR );
        exit( 1 );
    }

    // Initialize all pixels to black
    memset( *pLpvBits, 0, width * height * 4 );
}

float IntersectRaySphere ( _In_ const Vect3D origin,
                           _In_ const Vect3D direction_from_origin,
                           _In_ const Sphere sphere,
                           _In_ const float direction_from_origin_dot_product )
{
    /**
     * This is a simplified quadratic root equation
     * Mostly familiar in the form:
     * { t1, t2 } = (-b (+/-) sqrt(b^2 - 4ac) / 2a
     *
     * We just calculate one of the roots in favor of efficiency
     */

    Vect3D center_to_orign = origin - sphere.center;

    float a = direction_from_origin_dot_product;
    float b = 2 * center_to_orign.dot( direction_from_origin );
    float c = center_to_orign.dot( center_to_orign ) - sphere.raidus_squared;
    float discriminant = b * b - 4 * a * c;

    if ( discriminant < 0 )
    {
        return INFINITY;
    }
    else if ( discriminant == 0 )
    {
        return -b / ( 2 * a );
    }

    float t = ( -b - sqrtf( discriminant ) )
              / ( 2 * a ); // Minimize compute only go for 1 root;

    return t;
}

Vect3D CanvasToViewport ( _In_ const uint16_t x,
                          _In_ const uint16_t y,
                          _In_ const uint16_t width,
                          _In_ const uint16_t height )
{
    // for simplicity : Vw = Vh = d = 1    approx 53 fov
    float aspect_ratio = static_cast<float>( width ) / height;

    // Map x and y to the viewport, adjusting by aspect ratio
    float fov_mod = 1.0f;
    float viewport_x =
        ( x - width / 2.0f ) * ( ( 1.0f * fov_mod ) / width ) * aspect_ratio;
    float viewport_y = -( y - height / 2.0f )
                       * ( ( 1.0f * fov_mod )
                           / height ); // Flip Y to match 3D space orientation

    return Vect3D( viewport_x,
                   viewport_y,
                   1.0f ); // Z=1 for perspective projection
}

Intersection ClosestIntersection ( _In_ const Vect3D origin,
                                   _In_ const Vect3D direction_from_origin,
                                   _In_ const float t_min,
                                   _In_ const float t_max,
                                   _In_ std::vector<Sphere> &scene )
{
    Sphere *closest_sphere = NULL;
    float closest_point_t  = INFINITY;

    // Cache immutable value, saves compute
    float projection_plane_dot_product =
        direction_from_origin.dot( direction_from_origin );


    for ( auto &sphere : scene )
    {
        float possible_t = IntersectRaySphere( origin,
                                               direction_from_origin,
                                               sphere,
                                               projection_plane_dot_product );

        if ( IsBetterRoot( possible_t, t_min, t_max, closest_point_t ) )
        {
            closest_point_t = possible_t;
            closest_sphere  = const_cast<Sphere *>( &sphere );
        }
    }
    return Intersection( closest_sphere, closest_point_t );
}

WideColor TraceRay ( _In_ const Vect3D origin,
                     _In_ const Vect3D destination,
                     _In_ const float t_min,
                     _In_ const float t_max,
                     _In_ const uint8_t recursion_depth,
                     _In_ std::vector<Sphere> &scene,
                     _In_ std::vector<Light> &lights )
{
    Vect3D intersection_sphere_normal = { };
    Vect3D origin_to_destination      = { };
    Vect3D reflected_ray              = { };

    Intersection intersection =
        ClosestIntersection( origin, destination, t_min, t_max, scene );

    Sphere *closest_sphere           = intersection.sphere;
    float closest_intersection_point = intersection.point;

    /* No intersecting sphere = empty space */
    if ( closest_sphere == NULL )
    {
        return WideColor( 0, 0, 0 );
    }

    /* Compute intersection */
    origin_to_destination =
        origin + ( destination * closest_intersection_point );

    /* Computer sphere normal at intersection */
    intersection_sphere_normal =
        ( origin_to_destination - closest_sphere->center ).norm( );

    /* Calculate light modification to color of the point */
    float color_lighting_modifier = CalcLight( origin_to_destination,
                                               intersection_sphere_normal,
                                               destination.invert( ),
                                               closest_sphere->specularity,
                                               scene,
                                               lights );

    WideColor lit_color = ApplyMultiplierToColor( closest_sphere->color,
                                                  color_lighting_modifier );


    bool is_last_level_of_recursion = ( recursion_depth <= 0 );
    bool is_sphere_reflective       = ( closest_sphere->reflective <= 0 );

    /* Return if no reflections need to be calculated */
    if ( is_last_level_of_recursion || is_sphere_reflective )
    {
        return lit_color;
    }

    reflected_ray =
        ReflectRay( destination.invert( ), intersection_sphere_normal );

    /* Calculate color reflection */
    WideColor reflected_color = TraceRay( origin_to_destination,
                                          reflected_ray,
                                          t_min,
                                          t_max,
                                          recursion_depth - 1,
                                          scene,
                                          lights );


    WideColor final_color = CalculateFinalColor( lit_color,
                                                 reflected_color,
                                                 closest_sphere->reflective );

    return final_color;
}

void Draw ( _Inout_ BYTE **p_lpv_bits,
            _In_ const uint16_t width,
            _In_ const uint16_t height,
            _In_ Camera camera,
            _In_ std::vector<Sphere> &scene,
            _In_ std::vector<Light> &lights )
{
    Vect3D projection_plane_point  = { };
    Vect3D translated_camera_point = { };
    uint8_t recursionDepth         = 2;
    uint8_t bytes_in_a_pixel       = 4; /* Red, green, blue, and alpha */
    float t_min                    = 0.001f; /* Epsilon */
    float t_max                    = INFINITY;


    for ( uint16_t x = 0; x < width; ++x )
    {
        for ( uint16_t y = 0; y < height; ++y )
        {
            projection_plane_point = CanvasToViewport( x, y, width, height );
            translated_camera_point =
                camera.ApplyCameraRotation( projection_plane_point, camera );


            WideColor wide_color = TraceRay( camera.position,
                                             translated_camera_point,
                                             t_min,
                                             t_max,
                                             recursionDepth,
                                             scene,
                                             lights );

            COLORREF final_pixel_color = ClampColor( wide_color );
            /* canvas coordinates to memory pointer offset */
            uint32_t canvas_coordinate_offset =
                ( y * width + x ) * bytes_in_a_pixel;

            /* Max size of pixel buffer */
            uint32_t canvas_coordinate_upper_bound =
                width * height * bytes_in_a_pixel - bytes_in_a_pixel;

            /* Start of pixel buffer */
            uint8_t canvas_coordinate_lower_bound = 0;

            if ( IsInBounds( canvas_coordinate_offset,
                             canvas_coordinate_lower_bound,
                             canvas_coordinate_upper_bound ) )
            {
                SetPixelToColor( p_lpv_bits,
                                 canvas_coordinate_offset,
                                 final_pixel_color );
            }
        }
    }
}
