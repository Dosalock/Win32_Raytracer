#ifndef __HELPER_H__
#define __HELPER_H__
#pragma once

struct WideColor
{
    uint32_t red;
    uint32_t green;
    uint32_t blue;
};

/*---------------------Includes---------------------------*/
#include <Windows.h>
#include <algorithm>

/**
 * @brief Clamps color channel between 0 and 255
 *
 * @param color - Color channel
 *
 * @return Returns a uint16_t color channel thats been clamped
 */
inline COLORREF ClampColor ( WideColor color )
{
    uint8_t red   = std::clamp( color.red, 0u, 255u );
    uint8_t green = std::clamp( color.green, 0u, 255u );
    uint8_t blue  = std::clamp( color.blue, 0u, 255u );
    return RGB( red, green, blue );
}

/**
 * @brief Applies multiplier to color, channel wise
 *
 * @param[in] color - Color, COLOREF
 * @param[in] multiplier - Multiplier
 *
 * @return Color product of color and multiplier
 */
inline WideColor ApplyMultiplierToColor ( _In_ const COLORREF &color,
                                          _In_ const float &multiplier )
{
    uint32_t red   = static_cast<uint32_t>( GetRValue( color ) * multiplier );
    uint32_t green = static_cast<uint32_t>( GetGValue( color ) * multiplier );
    uint32_t blue  = static_cast<uint32_t>( GetBValue( color ) * multiplier );

    return WideColor( red, green, blue );
}

/**
 * @brief Calculates final color of point, taking reflectiveness and reflection
 * modifiers into account
 *
 * @param r - Red after light calculations
 * @param g - Green after light caluclations
 * @param b - Blue after light calculations
 * @param reflected_color - Color from inverted ray of reflection
 * @param reflectiveness - Reflectiveness of the sphere where the point exists
 *
 * @return
 */
inline WideColor CalculateFinalColor ( _In_ const WideColor &lit_color,
                                       _In_ const WideColor &reflected_color,
                                       _In_ const float &reflectiveness )
{
    float lit_multiplier = ( 1.0f - reflectiveness );

    // Calculate lit components (still unclamped)
    uint32_t lit_r = static_cast<uint32_t>( lit_color.red * lit_multiplier );
    uint32_t lit_g = static_cast<uint32_t>( lit_color.green * lit_multiplier );
    uint32_t lit_b = static_cast<uint32_t>( lit_color.blue * lit_multiplier );

    // Calculate reflected components (from reflected_color)
    uint32_t reflected_r =
        static_cast<uint32_t>( reflected_color.red * reflectiveness );
    uint32_t reflected_g =
        static_cast<uint32_t>( reflected_color.green * reflectiveness );
    uint32_t reflected_b =
        static_cast<uint32_t>( reflected_color.blue * reflectiveness );

    // Combine and clamp ONLY ONCE at the end
    return WideColor( lit_r + reflected_r,
                      lit_g + reflected_g,
                      lit_b + reflected_b );
}

/**
 * @brief Sets pixel at offset to color
 *
 * @param[in,out] p_lpv_bits - Pointer to pixel buffer
 * @param[in]     offset - Offset into pixel buffer for wanted pixel
 * @param[in]     color - Wanted color of pixel
 */
inline void SetPixelToColor ( _Inout_ BYTE **p_lpv_bits,
                              _In_ uint32_t &offset,
                              _In_ COLORREF &color )
{
    ( *p_lpv_bits )[offset + 0] = static_cast<uint32_t>( GetBValue( color ) );
    ( *p_lpv_bits )[offset + 1] = static_cast<uint32_t>( GetGValue( color ) );
    ( *p_lpv_bits )[offset + 2] = static_cast<uint32_t>( GetRValue( color ) );
    ( *p_lpv_bits )[offset + 3] = 255;
};

/*------------Template Declarations---------------*/
template<typename T>
concept Scalar = std::is_scalar_v<T>;

/**
 * @brief Returns true if a value is between, low and high
 *
 * @tparam T - Any scalar, std::is_scalar_v<T> == true
 *
 * @param value - value to check if its in bounds
 * @param low - lower bound
 * @param high - upper bound
 *
 * @return
 */
template<Scalar ValueType, Scalar LowType, Scalar HighType>
bool IsInBounds ( const ValueType &value,
                  const LowType &low,
                  const HighType &high )
{
    // Comparison logic will go here
    return !( value < low ) && ( value < high );
}

#endif // !__HELPER_H__
