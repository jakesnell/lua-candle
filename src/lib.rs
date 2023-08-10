use candle_core::{DType, Device, Error as CandleError, Tensor};
use mlua::prelude::*;
use mlua::FromLua;
use std::str::FromStr;

pub fn wrap_err(err: CandleError) -> LuaError {
    LuaError::RuntimeError(format!("{err:?}"))
}

struct LuaTensor(Tensor);

impl std::ops::Deref for LuaTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl LuaUserData for LuaTensor {
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, ()| {
            Ok(format!("{}", this.0))
        });
        methods.add_meta_function(
            LuaMetaMethod::Add,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaValue)| match rhs {
                LuaValue::UserData(ud) => {
                    let tensor = ud.borrow::<Self>()?;
                    Ok(LuaTensor((&lhs.0 + &tensor.0).map_err(wrap_err)?))
                }
                LuaValue::Integer(n) => Ok(LuaTensor((&lhs.0 + (n as f64)).map_err(wrap_err)?)),
                LuaValue::Number(n) => Ok(LuaTensor((&lhs.0 + n).map_err(wrap_err)?)),
                _ => unreachable!(),
            },
        );
        methods.add_meta_function(
            LuaMetaMethod::Sub,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaValue)| match rhs {
                LuaValue::UserData(ud) => {
                    let tensor = ud.borrow::<Self>()?;
                    Ok(LuaTensor((&lhs.0 - &tensor.0).map_err(wrap_err)?))
                }
                LuaValue::Integer(n) => Ok(LuaTensor((&lhs.0 - (n as f64)).map_err(wrap_err)?)),
                LuaValue::Number(n) => Ok(LuaTensor((&lhs.0 - n).map_err(wrap_err)?)),
                _ => unreachable!(),
            },
        );
        methods.add_meta_function(
            LuaMetaMethod::Mul,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaValue)| match rhs {
                LuaValue::UserData(ud) => {
                    let tensor = ud.borrow::<Self>()?;
                    Ok(LuaTensor((&lhs.0 * &tensor.0).map_err(wrap_err)?))
                }
                LuaValue::Integer(n) => Ok(LuaTensor((&lhs.0 * (n as f64)).map_err(wrap_err)?)),
                LuaValue::Number(n) => Ok(LuaTensor((&lhs.0 * n).map_err(wrap_err)?)),
                _ => unreachable!(),
            },
        );
        methods.add_meta_function(
            LuaMetaMethod::Div,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaValue)| match rhs {
                LuaValue::UserData(ud) => {
                    let tensor = ud.borrow::<Self>()?;
                    Ok(LuaTensor((&lhs.0 / &tensor.0).map_err(wrap_err)?))
                }
                LuaValue::Integer(n) => Ok(LuaTensor((&lhs.0 / (n as f64)).map_err(wrap_err)?)),
                LuaValue::Number(n) => Ok(LuaTensor((&lhs.0 / n).map_err(wrap_err)?)),
                _ => unreachable!(),
            },
        );
        methods.add_method("sum_all", |_, this, ()| {
            Ok(LuaTensor(this.sum_all().map_err(wrap_err)?))
        });
        methods.add_method("to", |_, this, dtype: LuaDType| {
            Ok(LuaTensor(this.to_dtype(dtype.0).map_err(wrap_err)?))
        });
        methods.add_method("matmul", |_, this, other: LuaUserDataRef<Self>| {
            Ok(LuaTensor(this.matmul(&other).map_err(wrap_err)?))
        });
        methods.add_method("reshape", |_, this, shape: Vec<usize>| {
            Ok(LuaTensor(this.reshape(shape).map_err(wrap_err)?))
        });
        methods.add_method("shape", |_, this, ()| -> LuaResult<Vec<usize>> {
            Ok(this.dims().to_vec())
        })
    }
}

#[derive(Copy, Clone)]
struct LuaDType(DType);

impl LuaUserData for LuaDType {
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, ()| {
            Ok(format!("{:?}", this.0))
        });
        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, ()| {
            Ok(format!("{:?}", this.0))
        });
    }
}

impl<'lua> FromLua<'lua> for LuaDType {
    fn from_lua(value: LuaValue<'lua>, _: &'lua Lua) -> LuaResult<Self> {
        match value {
            LuaValue::UserData(ud) => Ok(*ud.borrow::<Self>()?),
            LuaValue::String(dtype_luastr) => {
                let dtype_str = dtype_luastr.to_str()?;
                let dtype = DType::from_str(dtype_str)
                    .map_err(|_| LuaError::RuntimeError(format!("invalid dtype '{dtype_str}'")))?;
                Ok(Self(dtype))
            }
            LuaValue::Nil => Ok(Self(DType::F64)),
            _ => unreachable!(),
        }
    }
}

fn new_tensor(_: &Lua, value: f64) -> LuaResult<LuaTensor> {
    let tensor = Tensor::new(value, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

fn ones(_: &Lua, (shape, dtype): (Vec<usize>, LuaDType)) -> LuaResult<LuaTensor> {
    let tensor = Tensor::ones(shape, dtype.0, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

fn zeros(_: &Lua, (shape, dtype): (Vec<usize>, LuaDType)) -> LuaResult<LuaTensor> {
    let tensor = Tensor::zeros(shape, dtype.0, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

fn rand(_: &Lua, shape: Vec<usize>) -> LuaResult<LuaTensor> {
    let tensor = Tensor::rand(0f64, 1f64, shape, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

fn randn(_: &Lua, shape: Vec<usize>) -> LuaResult<LuaTensor> {
    let tensor = Tensor::randn(0f64, 1f64, shape, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

#[mlua::lua_module]
fn candle(lua: &Lua) -> LuaResult<LuaTable> {
    let exports = lua.create_table()?;
    exports.set("Tensor", lua.create_function(new_tensor)?)?;
    exports.set("ones", lua.create_function(ones)?)?;
    exports.set("zeros", lua.create_function(zeros)?)?;
    exports.set("rand", lua.create_function(rand)?)?;
    exports.set("randn", lua.create_function(randn)?)?;
    exports.set("u8", LuaDType(DType::U8))?;
    exports.set("u32", LuaDType(DType::U32))?;
    exports.set("bf16", LuaDType(DType::BF16))?;
    exports.set("f16", LuaDType(DType::F16))?;
    exports.set("f32", LuaDType(DType::F32))?;
    exports.set("f64", LuaDType(DType::F64))?;

    Ok(exports)
}
