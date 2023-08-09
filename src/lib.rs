use candle_core::{DType, Device, Error as CandleError, Tensor};
use mlua::prelude::*;

pub fn wrap_err(err: CandleError) -> LuaError {
    LuaError::RuntimeError(format!("{err:?}"))
}

struct LuaTensor(Tensor);

impl LuaUserData for LuaTensor {
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_meta_method(LuaMetaMethod::Call, |_, _, value: f32| {
            let tensor = Tensor::new(value, &Device::Cpu).map_err(wrap_err)?;
            Ok(LuaTensor(tensor))
        });
        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, ()| {
            Ok(format!("{}", this.0))
        });
        methods.add_meta_function(
            LuaMetaMethod::Add,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaUserDataRef<Self>)| {
                Ok(LuaTensor(Tensor::add(&lhs.0, &rhs.0).map_err(wrap_err)?))
            },
        );
        methods.add_meta_function(
            LuaMetaMethod::Sub,
            |_, (lhs, rhs): (LuaUserDataRef<Self>, LuaUserDataRef<Self>)| {
                Ok(LuaTensor(Tensor::sub(&lhs.0, &rhs.0).map_err(wrap_err)?))
            },
        );
    }
}

fn ones(_: &Lua, shape: Vec<usize>) -> LuaResult<LuaTensor> {
    let tensor = Tensor::ones(shape, DType::F32, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

fn zeros(_: &Lua, shape: Vec<usize>) -> LuaResult<LuaTensor> {
    let tensor = Tensor::zeros(shape, DType::F32, &Device::Cpu).map_err(wrap_err)?;
    Ok(LuaTensor(tensor))
}

#[mlua::lua_module]
fn candle(lua: &Lua) -> LuaResult<LuaTable> {
    let exports = lua.create_table()?;
    exports.set("ones", lua.create_function(ones)?)?;
    exports.set("zeros", lua.create_function(zeros)?)?;
    exports.set(
        "Tensor",
        lua.create_userdata(LuaTensor(
            Tensor::zeros((), DType::F32, &Device::Cpu).map_err(wrap_err)?,
        ))?,
    )?;

    Ok(exports)
}
